module Sum 

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils

let sumOfArray (context: OpenCLEvaluationContext) (array: int[]) = 
    let localSize = 
        [
            256
            (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
        ] |> List.min

    let result = Array.zeroCreate 1
    let resizedArray = Array.zeroCreate <| getMultipleSize array.Length localSize
    Array.blit array 0 resizedArray 0 array.Length

    let kernel = 
        <@
            fun (ndRange: _1D)
                (array: int[])
                (result: int[]) ->

                let localId = ndRange.LocalID0
                let globalId = ndRange.GlobalID0

                let localBuffer = localArray<int> localSize
                localBuffer.[localId] <- array.[globalId]
                barrier ()

                let mutable amountOfValuesToSum = localSize
                while amountOfValuesToSum > 1 do 
                    if localId * 2 < amountOfValuesToSum then
                        let a = localBuffer.[localId]
                        let b = localBuffer.[localId + amountOfValuesToSum / 2]
                        localBuffer.[localId] <- a + b
                    amountOfValuesToSum <- amountOfValuesToSum / 2
                    barrier ()

                if localId = 0 then 
                    // this should be atomic <!+
                    result.[0] <- result.[0] + localBuffer.[localId] 
        @>

    let ndRange = _1D(resizedArray.Length, localSize)
    let binder = fun kernelPrepare ->
        kernelPrepare
            ndRange
            resizedArray
            result

    opencl {
        do! RunCommand kernel binder
        return! ToHost result
    }
    |> context.RunSync
    |> (fun f x y -> f y x) Array.get 0

// В РАНТАЙМЕ не узнать размер рабочей группы тк не поддерживается :>
// атомарное сложение не хочет рабоать <!+ (не транслируется)
// значения из замыкания инлайнятся?
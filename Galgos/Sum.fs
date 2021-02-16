module Sum 

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils

let sumOfArray (array: int[]) = 
    opencl {
        let! context = getEvaluationContext
        let localSize = 
            [
                256
                (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
            ] |> List.min

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
                        result.[0] <!+ localBuffer.[localId] 
            @>

        let result = Array.zeroCreate 1
        let resizedArray = Array.zeroCreate <| getMultipleSize localSize array.Length
        Array.blit array 0 resizedArray 0 array.Length

        do! RunCommand kernel (fun kernelPrepare ->
            let ndRange = _1D(resizedArray.Length, localSize)
            kernelPrepare
                ndRange
                resizedArray
                result
        )

        return result
    }
// В РАНТАЙМЕ не узнать размер рабочей группы тк не поддерживается :>
// атомарное сложение не хочет рабоать <!+ (не транслируется)
// значения из замыкания инлайнятся?
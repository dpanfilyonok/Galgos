module PrefixSum 

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils

let prefixSum (context: OpenCLEvaluationContext) (array: int[]) = 
    let localSize = 
        [
            256
            (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
        ] |> List.min

    let resizedArray = Array.zeroCreate <| getMultipleSize array.Length localSize
    Array.blit array 0 resizedArray 0 array.Length

    let globalSize = resizedArray.Length
    let groupCount = globalSize / localSize
    let chunkSums = Array.zeroCreate<int> groupCount

    let kernel = 
        <@
            fun (ndRange: _1D)
                (array: int[])
                (chunkSums: int[]) ->

                let globalId = ndRange.GlobalID0
                let localId = ndRange.LocalID0
                let groupId = globalId / localSize

                let localBuffer = localArray<int> localSize
                localBuffer.[localId] <- array.[globalId]
                barrier ()
                
                if localId = 0 then
                    for i in 1 .. localSize - 1 do  
                        localBuffer.[i] <- localBuffer.[i] + localBuffer.[i - 1]
                    chunkSums.[groupId] <- localBuffer.[localSize - 1]
                barrier ()

                array.[globalId] <- localBuffer.[localId]
                barrier ()

                if groupId * 2 < groupCount then
                    let height = int (System.Math.Log(float groupCount, float 2))
                    for i in 1 .. height do
                        // определеят размер чанков на каждой итерации
                        let chunkSize = int (System.Math.Pow(2., float i))
                        // определеят, в какой чанк попадает рабочая группа
                        let chunkId = groupId * 2 / chunkSize
                        // определеят смещение начала чанка в рабочих группах
                        let chunkOffset = chunkId * chunkSize
                        // определяет смещение группы, куда должна производить апись текущая группа 
                        let groupOffset = chunkOffset + chunkSize / 2 + groupId % (chunkSize / 2)
                        let step = chunkSums.[chunkId + chunkSize / 2 - 1]

                        localBuffer.[localId] <- array.[groupOffset * localSize + localId]
                        localBuffer.[localId] <- step + localBuffer.[localId]
                        array.[groupOffset * localSize + localId] <- localBuffer.[localId]
                        chunkSums.[groupOffset] <- localBuffer.[localSize - 1]
                        barrier ()
        @>

    let ndRange = _1D(resizedArray.Length, localSize)
    let binder = fun kernelPrepare ->
        kernelPrepare
            ndRange
            resizedArray
            chunkSums

    opencl {
        do! RunCommand kernel binder
        return! ToHost resizedArray
    }
    |> context.RunSync
    |> (fun result -> result.[0 .. array.Length - 1])

// логарифм по основанию 2 не поддерживается
// пайп оператор не поддерживается
// обычный лог принимает только 1 аргумент
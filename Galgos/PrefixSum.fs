module PrefixSum 

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils
open Brahma.FSharp.OpenCL.Core

let prefixSum (array: int[]) = 
    opencl {
        let! context = getEvaluationContext
        let workGroupSize = 
            [
                256
                (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
            ] |> List.min

        let newArraySize = getMultipleSize workGroupSize array.Length 
        let globalGroupCount = newArraySize / workGroupSize
        let height = System.Math.Log2(float globalGroupCount) |> int

        let kernel1 = 
            <@
                fun (ndRange: _1D)
                    (array: int[])
                    (chunkSums: int[]) ->

                    let globalId = ndRange.GlobalID0
                    let localId = ndRange.LocalID0
                    let groupId = globalId / workGroupSize

                    let localBuffer = localArray<int> workGroupSize
                    localBuffer.[localId] <- array.[globalId]
                    barrier ()
                    
                    if localId = 0 then
                        for i in 1 .. workGroupSize - 1 do  
                            localBuffer.[i] <- localBuffer.[i] + localBuffer.[i - 1]
                        chunkSums.[groupId] <- localBuffer.[workGroupSize - 1]
                    barrier ()

                    array.[globalId] <- localBuffer.[localId]
                    barrier ()
            @>

        let kernel2 = 
            <@
                fun (ndRange: _1D)
                    (array: int[])
                    (chunkSums: int[]) ->

                    let globalId = ndRange.GlobalID0
                    let localId = ndRange.LocalID0
                    let groupId = globalId / workGroupSize

                    let localBuffer = localArray<int> workGroupSize
               
                    for i in 1 .. height do
                        // определеят размер чанков на каждой итерации
                        let chunkSize = int (System.Math.Pow(2., float i))
                        // определеят, в какой чанк попадает рабочая группа
                        let chunkId = groupId * 2 / chunkSize
                        // определеят смещение начала чанка в рабочих группах
                        let chunkOffset = chunkId * chunkSize
                        // определяет смещение группы, куда должна производить запись текущая группа 
                        let groupOffset = chunkOffset + chunkSize / 2 + groupId % (chunkSize / 2)
                        let step = chunkSums.[chunkOffset + chunkSize / 2 - 1]

                        localBuffer.[localId] <- array.[groupOffset * workGroupSize + localId]
                        barrier ()
                        if i = 2 then 
                            localBuffer.[localId] <- localBuffer.[localId]
                        else
                            localBuffer.[localId] <- localBuffer.[localId] + step
                        barrier ()
                        array.[groupOffset * workGroupSize + localId] <- localBuffer.[localId]
                        barrier ()
                        if localId = 0 then
                            chunkSums.[groupOffset] <- localBuffer.[workGroupSize - 1]
                        barrier ()
            @>

        let resizedArray = Array.zeroCreate<int> newArraySize
        Array.blit array 0 resizedArray 0 array.Length
        let chunkSums = Array.zeroCreate<int> globalGroupCount

        do! RunCommand kernel1 (fun kernelPrepare ->
            let ndRange = _1D(resizedArray.Length, workGroupSize)
            kernelPrepare
                ndRange
                resizedArray
                chunkSums
        )

        do! RunCommand kernel2 (fun kernelPrepare ->
            let ndRange = _1D(resizedArray.Length / 2, workGroupSize)
            kernelPrepare
                ndRange
                resizedArray
                chunkSums
        )

        // let! ra = ToHost resizedArray
        // let arr = Array.zeroCreate<int> array.Length
        // Array.blit ra 0 arr 0 array.Length
        // return arr
        return! ToHost chunkSums
    }


// логарифм по основанию 2 не поддерживается
// пайп оператор не поддерживается
// обычный лог принимает только 1 аргумент
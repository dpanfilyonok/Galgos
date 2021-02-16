module Filter

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils
open FSharp.Quotations

let filter (context: OpenCLEvaluationContext) (array: int[]) (predicate: Expr<int -> bool>) = 
    // Устанавливаем глобальный и локальный размер
    let localSize = 
        [
            256
            (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
        ] |> List.min
    let globalSize = getMultipleSize localSize array.Length
    let groupCount = globalSize / localSize
    let filteredArraySize = globalSize / 4

    // Описываем кернел
    let kernel = 
        <@
            fun (ndRange: _1D)
                (array: int[])
                (filteredArray: int[])
                (allocatedMem: int) ->

                let localId = ndRange.LocalID0
                let globalId = ndRange.GlobalID0
                
                let localBuffer = localArray<int> localSize
                localBuffer.[localId] <- array.[globalId]
                barrier ()

                let mutable filteredCount = local<int> ()
                if localId = 0 then 
                    filteredCount <- 0
                barrier()

                let mutable idx = -1
                if (%predicate) localBuffer.[localId] then
                    idx <- aIncrR filteredCount
                barrier ()

                let localFilteredArray = localArray<int> filteredCount
                if idx <> -1 then
                    localFilteredArray.[idx] <- localBuffer.[localId]
                barrier ()

                let mutable globalOffset = local<int> ()
                if localId = 0 then
                    globalOffset <- allocatedMem <!+> filteredCount
                barrier ()

                if localId < filteredCount then
                    filteredArray.[globalOffset + localId] <- localFilteredArray.[localId]
                barrier ()
        @>
    
    // Описываем входные данные для кернела
    let resizedArray = Array.zeroCreate globalSize
    Array.blit array 0 resizedArray 0 array.Length
    let filteredArray = Array.zeroCreate filteredArraySize
    let allocatedMem = 0
    
    // Описываем ndRange и binder
    let ndRange = _1D(globalSize, localSize)
    let binder = fun kernelPrepare ->
        kernelPrepare
            ndRange
            resizedArray
            filteredArray
            allocatedMem
    
    // Запускаем вычисления
    opencl {
        do! RunCommand kernel binder
        return! ToHost filteredArray
    }
    |> context.RunSync
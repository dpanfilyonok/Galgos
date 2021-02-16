module MergeSorted

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open FSharp.Quotations

let merge (left: int[]) (right: int[]) = 
    opencl {
        let! context = getEvaluationContext
        let workGroupSize = 
            [
                256
                (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
            ] |> List.min
        let mergedLen = left.Length + right.Length
        let resizedMergedLen = Utils.getMultipleSize workGroupSize mergedLen 
        let iterationCount = resizedMergedLen / workGroupSize
        
        let f = 
            <@
                fun (globalId: int) (x: int) -> 
                    let i = globalId - x
                    let j = x
                    (i, j)
            @>

        // Описываем кернел
        let kernel1 = 
            <@
                fun (ndRange: _1D)
                    (leftArray: int[])
                    (rightArray: int[])
                    (merged: int[]) ->

                    let globalId = ndRange.GlobalID0

                    let dim0 = localArray<int> workGroupSize
                    let dim1 = localArray<int> workGroupSize
                    let mergedPart = localArray<int> workGroupSize

                    let mutable partStartI = local<int> ()
                    let mutable partStartJ = local<int> ()
                    partStartI <- 0
                    partStartJ <- 0

                    // let f = fun x -> 
                    //     let i = globalId - x
                    //     let j = x
                    //     (i, j)

                    let currentIteration = 0
                    while currentIteration < iterationCount do
                        dim0.[globalId] <- leftArray.[partStartI + globalId]
                        dim1.[globalId] <- rightArray.[partStartJ + globalId]
                        barrier ()

                        let mutable leftIdx = 0
                        let mutable rightIdx = globalId
                        while rightIdx - leftIdx > 1 do
                            let middle = leftIdx + (rightIdx - leftIdx) / 2
                            let (middleI, middleJ) = (%f) globalId middle
                            if dim0.[middleI] <= dim1.[middleJ] then
                                rightIdx <- middle
                            else 
                                leftIdx <- middle
                            
                        let (leftI, leftJ) = (%f) globalId leftIdx
                        let (rightI, rightJ) = (%f) globalId rightIdx

                        // левый верхний квадрат
                        if leftIdx = rightIdx then
                            if dim0.[leftI] <= dim1.[rightJ] then
                                mergedPart.[globalId] <- dim0.[leftI]
                            else 
                                mergedPart.[globalId] <- dim1.[leftJ]

                        // 0|0
                        // 0|0       
                        elif dim0.[rightI] <= dim1.[rightJ] then
                            mergedPart.[globalId] <- dim0.[leftI]

                        // 1|1
                        // 1|1
                        elif dim0.[leftI] > dim1.[leftJ] then
                            mergedPart.[globalId] <- dim1.[rightJ]

                        else
                            if dim0.[rightI] <= dim1.[leftJ] then
                                mergedPart.[globalId] <- dim1.[leftJ]
                            else 
                                mergedPart.[globalId] <- dim0.[rightI]

                        if globalId = workGroupSize - 1 then
                            partStartI <- dim0.[leftI]
                            partStartJ <- dim1.[rightJ]
                        barrier () // ???

                        merged.[currentIteration * workGroupSize + globalId] <- mergedPart.[globalId]
                        barrier () // ???
            @>

        // Описываем входные данные для кернела
        let diffLen = resizedMergedLen - mergedLen
        let resizedLeft = Array.create<int> (left.Length + diffLen + workGroupSize / 2) System.Int32.MaxValue
        Array.blit left 0 resizedLeft 0 left.Length
        let resizedRight = Array.create<int> (left.Length + workGroupSize / 2) System.Int32.MaxValue
        Array.blit right 0 resizedRight 0 right.Length
        let resizedMerged = Array.zeroCreate<int> resizedMergedLen
        
        // Запускаем вычисления
        do! RunCommand kernel1 (fun kernelPrepare ->
            let ndRange = _1D(workGroupSize, workGroupSize)
            kernelPrepare
                ndRange
                resizedMerged
                resizedLeft
                resizedRight
        )

        return! ToHost resizedMerged
    }
    
    // let a = left.Length
    // let b = right.Length

    // let kernel2 = 
    //     <@
    //         fun (ndRange: _1D)
    //             (merged: int[])
    //             (leftArray: int[])
    //             (rightArray: int[]) ->

    //             let globalId = ndRange.GlobalID0
                
    //             if globalId = 0 then
    //                 let mutable leftPtr = 0
    //                 let mutable rightPtr = 0
    //                 for i in 0 .. a + b - 1 do
    //                     if leftPtr = a then 
    //                         merged.[i] <- rightArray.[rightPtr]
    //                         rightPtr <- rightPtr + 1
    //                     elif rightPtr = b then
    //                         merged.[i] <- leftArray.[leftPtr]
    //                         leftPtr <- leftPtr + 1
    //                     elif leftArray.[leftPtr] <= rightArray.[rightPtr] then 
    //                         merged.[i] <- leftArray.[leftPtr]
    //                         leftPtr <- leftPtr + 1
    //                     else 
    //                         merged.[i] <- rightArray.[rightPtr]
    //                         rightPtr <- rightPtr + 1
    //     @>

type COOFormat<'a> = { 
    Rows: int []
    Columns: int []
    Values: 'a []
    RowCount: int
    ColumnCount: int 
}

let eWiseAddCOO (context: OpenCLEvaluationContext) (left: COOFormat<int>) (right: COOFormat<int>) = 
    let columnCount = left.ColumnCount
    let k1 = 
        <@
            fun (ndRange: _1D)
                (leftRows: int[])
                (leftColumns: int[])
                (rightRows: int[])
                (rightColumns: int[])
                (leftIdx: int[])
                (rightIdx: int[]) ->

                let globalId = ndRange.GlobalID0
                leftIdx.[globalId] <- leftRows.[globalId] * columnCount + leftColumns.[globalId]
                rightIdx.[globalId] <- rightRows.[globalId] * columnCount + rightColumns.[globalId]
        @>

    let a = left.Values.Length
    let b = right.Values.Length
    let k2 = 
        <@
            fun (ndRange: _1D)
                (leftIdx: int[])
                (leftValues: int[])
                (rightIdx: int[])
                (rightValues: int[])
                (mergedIdx: int[])
                (mergedValues: int[]) ->

                let globalId = ndRange.GlobalID0
                
                if globalId = 0 then
                    let mutable leftPtr = 0
                    let mutable rightPtr = 0
                    for i in 0 .. a + b - 1 do
                        if leftPtr = a then 
                            mergedIdx.[i] <- rightIdx.[rightPtr]
                            mergedValues.[i] <- rightValues.[rightPtr]
                            rightPtr <- rightPtr + 1
                        elif rightPtr = b then
                            mergedIdx.[i] <- leftIdx.[leftPtr]
                            mergedValues.[i] <- leftValues.[leftPtr]
                            leftPtr <- leftPtr + 1
                        elif leftIdx.[leftPtr] <= rightIdx.[rightPtr] then 
                            mergedIdx.[i] <- leftIdx.[leftPtr]
                            mergedValues.[i] <- leftValues.[leftPtr]
                            leftPtr <- leftPtr + 1
                        else 
                            mergedIdx.[i] <- rightIdx.[rightPtr]
                            mergedValues.[i] <- rightValues.[rightPtr]
                            rightPtr <- rightPtr + 1
        @>

    let k3 = 
        <@
            fun (ndRange: _1D)
                (mergedIdx: int[]) 
                (isUnique: int[]) ->

                let globalId = ndRange.GlobalID0
                if globalId = 0 then
                    isUnique.[globalId] <- 1
                else 
                    let prev = mergedIdx.[globalId - 1]
                    let curr = mergedIdx.[globalId]
                    if curr <> prev then 
                        isUnique.[globalId] <- 1
        @>

    let k4 = 
        <@
            fun (ndRange: _1D)
                (isUnique: int[])
                (position: int[])
                (len: int[]) ->

                let globalId = ndRange.GlobalID0
                if globalId = 0 then
                    let mutable acc = 0 
                    for i in 0 .. a + b - 1 do
                        let value = isUnique.[i]
                        acc <- acc + value
                        position.[i] <- acc
                    len.[0] <- acc
        @>
    
    let k5 = 
        <@
            fun (ndRange: _1D)
                (mergedIdx: int[])
                (mergedValues: int[])
                (position: int[])
                (resultRows: int[])
                (resultCols: int[])
                (resultValues: int[]) ->

                let globalId = ndRange.GlobalID0
                let position = position.[globalId]
                resultValues.[position] <- resultValues.[position] + mergedValues.[globalId]

                let idx = mergedIdx.[globalId]
                let rowIdx = idx / columnCount
                let colIdx = idx % columnCount
                resultRows.[position] <- rowIdx
                resultCols.[position] <- colIdx
        @>
    
    let leftIdx = Array.zeroCreate<int> a
    let rightIdx = Array.zeroCreate<int> b

    let mergedIdx = Array.zeroCreate<int> (a + b)
    let mergedValues = Array.zeroCreate<int> (a + b)

    let isUnique = Array.zeroCreate<int> (a + b)

    let position = Array.zeroCreate<int> (a + b)

    let len = [|0|]

    opencl {
        do! RunCommand k1 (fun kernelPrepare ->
            kernelPrepare
                (_1D <| columnCount)
                left.Rows
                left.Columns
                right.Rows
                right.Columns
                leftIdx
                rightIdx
        )
        
        do! RunCommand k2 (fun kernelPrepare ->
            kernelPrepare
                (_1D <| 64)
                leftIdx
                left.Values
                rightIdx
                right.Values
                mergedIdx
                mergedValues
        )
         
        do! RunCommand k3 (fun kernelPrepare ->
            kernelPrepare
                (_1D <| a + b)
                mergedIdx
                isUnique
        )

        do! RunCommand k4 (fun kernelPrepare ->
            kernelPrepare
                (_1D <| 64)
                isUnique
                position
                len
        )

        let! lenAr = (ToHost len)
        let len = lenAr.[0]

        let resultRows = Array.zeroCreate<int> len
        let resultCols = Array.zeroCreate<int> len
        let resultValues = Array.zeroCreate<int> len

        do! RunCommand k5 (fun kernelPrepare ->
            kernelPrepare
                (_1D <| a + b)
                mergedIdx
                mergedValues
                position
                resultRows
                resultCols
                resultValues
        )

        let! a = ToHost resultRows
        let! b = ToHost resultCols
        let! c = ToHost resultValues
        return (a, b, c)
    }
    |> context.RunSync

type CSRFormat<'a> = {
    Values: 'a[]
    Columns: int[]
    RowPointers: int[]
    ColumnCount: int
}

let eWiseAddCSR (left: CSRFormat<float>) (right: CSRFormat<float>) = 
    let wgSize1 = 1
    let k1 = 
        <@
            fun (ndRange: _1D)
                (ptr: int[])
                (ptrLen: int)
                (cols: int[]) 
                (ncols: int)
                (idx: int[]) ->

                let globalId = ndRange.GlobalID0
                let localId = ndRange.LocalID0

                let mutable localPtrLen = local<int> ()
                localPtrLen <- ptrLen
                
                let mutable localNcols = local<int> ()
                localNcols <- ncols

                let localAllPtr = localArray<int> localPtrLen
                // копируем массив в локальную память для каждой рабочей группы
                barrier ()

                let localIdx = localArray<int> wgSize1
                localIdx.[localId] <- 0

                let colIdx = cols.[globalId]

                let nz = globalId + 1
                let mutable currentRow = 0
                while localAllPtr.[currentRow + 1] < nz do
                    currentRow <- currentRow + 1

                localIdx.[localId] <- currentRow * localNcols + colIdx
                idx.[globalId] <- localIdx.[localId]
        @>

    let a = left.Values.Length
    let b = right.Values.Length

    // let kernel1 = 
    //     <@
    //         fun (ndRange: _1D)
    //             (leftArray: int[])
    //             (rightArray: int[])
    //             (merged: int[]) ->

    //             let localId = ndRange.LocalID0

    //             let mutable partStartI = local<int> ()
    //             let mutable partStartJ = local<int> ()
    //             partStartI <- 0
    //             partStartJ <- 0

    //             let leftPart = localArray<int> tileSize
    //             let rightPart = localArray<int> tileSize

    //             while true do
    //                 leftPart.[localId] <- leftArray.[localId + partStartI]
    //                 rightPart.[localId] <- rightArray.[localId + partStartJ]
    //                 let mutable leftPtr = localId / 2
    //                 let mutable rightPtr = localId / 2
    //                 for _ in 0 .. int (System.Math.Log2(float localId)) - 1 do 
    //                     if leftPart.[leftPtr] < rightPart.[rightPtr] then
    //                         leftPtr <- leftPtr - leftPtr / 2
    //                         rightPtr <- rightPtr + rightPtr / 2
    //                     else 
    //                         leftPtr <- leftPtr + leftPtr / 2
    //                         rightPtr <- rightPtr - rightPtr / 2
    //             ()
    //     @>

    ()

// было бе неплохо аргументом local передавать значение, которым нужно инициализовать переменную 
// -- не, так нельзя, тк непонятно, какой поток переменную инициализирует

// нельзя прям внутри кернела функцию написать
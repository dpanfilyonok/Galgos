module MatrixTranspose

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils
open FSharp.Quotations

let transposeMatrix (context: OpenCLEvaluationContext) (matrix: int[,]) = 
    // Устанавливаем глобальный и локальный размер
    let tileSize = 
        [
            256
            (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
        ] 
        |> List.min 
        |> float |> sqrt |> int
    let tileSize2 = tileSize * tileSize
    let globalSizeX = getMultipleSize tileSize (Array2D.length2 matrix)
    let globalSizeY = getMultipleSize tileSize (Array2D.length1 matrix) 
    
    // Описываем кернел
    let kernel = 
        <@
            fun (ndRange: _2D)
                (matrix: int[,])
                (transposed: int[]) ->

                let localI = ndRange.LocalID1
                let localJ = ndRange.LocalID0
                let globalI = ndRange.GlobalID1
                let globalJ = ndRange.GlobalID0

                let localTile = localArray<int> (5 * 5)
                localTile.[localJ * tileSize + localI] <- matrix.[globalI, globalJ]
                barrier ()

                if localI >= localJ then    
                    let temp = localTile.[localI * tileSize + localJ]
                    localTile.[localI * tileSize + localJ] <- localTile.[localJ * tileSize + localI]
                    localTile.[localJ * tileSize + localI] <- temp
                barrier ()

                transposed.[globalJ * globalSizeX + globalI] <- localTile.[localJ * tileSize + localI]
                barrier ()
        @>
    
    // Описываем входные данные для кернела
    let resizedMatrix = Array2D.zeroCreate<int> globalSizeY globalSizeX
    Array2D.blit matrix 0 0 resizedMatrix 0 0 (Array2D.length1 matrix) (Array2D.length2 matrix)
    let transposedMatrix = Array.zeroCreate<int> (globalSizeX * globalSizeY)
    
    // Описываем ndRange и binder
    let ndRange = _2D(globalSizeX, globalSizeY, tileSize, tileSize)
    
    // Запускаем вычисления
    opencl {
        do! RunCommand kernel (fun kernelPrepare ->
            kernelPrepare
                ndRange
                resizedMatrix
                transposedMatrix
            )
        return! ToHost transposedMatrix
    }
    |> context.RunSync
    |> (fun arr -> Array2D.init globalSizeX globalSizeY (fun i j -> arr.[i * globalSizeX + j]))

// проблема, что непонятно какая ось имеется ввиду (0 и 1)
// каждый раз приходится вручную заполнять все индексы -- можно сделать метод, который возвращал бы их кортеж или рекорд
// почему мы вообще ndRange в каечстве параметра кернела передаем
// localArray принимает не константы объявленные только вне кернела, 5*5 он не принимает
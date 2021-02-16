module ParticleInteraction

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Utils
open FSharp.Quotations

// f(a, a) = 0
// f(a, 0) = f(0, a) = 0
let particleInteraction (context: OpenCLEvaluationContext) (particlesArray: int[]) (f: Expr<int -> int -> int>) =
    let localSize = 
        [
            256
            (Cl.GetDeviceInfo(context.Device, DeviceInfo.MaxWorkGroupSize) |> fst).CastTo<int>()
        ] |> List.min
    let globalSize = getMultipleSize localSize particlesArray.Length 

    let resizedArray = Array.zeroCreate globalSize
    Array.blit particlesArray 0 resizedArray 0 particlesArray.Length

    // каждая рабочая группа обрабатывает свою вертикальную полосу
    let kernel = 
        <@
            fun (ndRange: _1D)
                (particles: int[]) ->

                let localId = ndRange.LocalID0
                let globalId = ndRange.GlobalID0

                let interactionAccumulator = localArray<int> localSize
                let localParticles = localArray<int> localSize
                localParticles.[localId] <- particles.[globalId]
                barrier ()

                let processedParticles = localArray<int> localSize
                for i in 0 .. globalSize / localSize - 1 do
                    processedParticles.[localId] <- particles.[localSize * i + localId]
                    barrier ()

                    for i in 0 .. localSize - 1 do  
                        interactionAccumulator.[localId] <-
                            interactionAccumulator.[localId] + 
                            (%f) localParticles.[localId] processedParticles.[i]
                    barrier ()

                localParticles.[localId] <- localParticles.[localId] + interactionAccumulator.[localId]
                particles.[globalId] <- localParticles.[localId]
                barrier()
        @>
    
    let ndRange = _1D(globalSize, localSize)
    let binder = fun kernelPrepare ->
        kernelPrepare
            ndRange
            resizedArray

    opencl {
        do! RunCommand kernel binder
        return! ToHost resizedArray
    }
    |> context.RunSync
    |> (fun array -> array.[0 .. particlesArray.Length - 1])
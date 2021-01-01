open System

open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open OpenCL.Net
open Galgos

[<EntryPoint>]
let main argv =
    let context = OpenCLEvaluationContext("INTEL*", DeviceType.Cpu)
    printfn "%i" <| sum context (Array.init 5 id)
    0
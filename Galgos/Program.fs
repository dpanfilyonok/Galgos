open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open OpenCL.Net
open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Sum
open PrefixSum

[<EntryPoint>]
let main argv =
    let context = OpenCLEvaluationContext("INTEL*", DeviceType.Cpu)
    printfn "%A" <| prefixSum context (Array.init 512 id) 
    0
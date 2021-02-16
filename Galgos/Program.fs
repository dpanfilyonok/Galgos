open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open OpenCL.Net
open Brahma.FSharp.OpenCL.WorkflowBuilder.Evaluation
open Brahma.FSharp.OpenCL.WorkflowBuilder.Basic
open Brahma.OpenCL
open OpenCL.Net
open Sum
open PrefixSum
open MatrixTranspose
open MergeSorted
open Brahma.FSharp.OpenCL.Core

[<EntryPoint>]
let main argv =
    let context =
        OpenCLEvaluationContext("INTEL*", DeviceType.Cpu)

    let array = Array.init 1024 id
    opencl { 
        let! sum = sumOfArray array 
        let! hostSum = ToHost sum
        return hostSum.[0]
    }
    |> context.RunSync
    |> printfn "%i"
    
    0
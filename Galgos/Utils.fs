module Utils 

let getMultipleSize originalSize multiplicity = 
    if originalSize % multiplicity = 0 then 
        originalSize
    else 
        (originalSize / multiplicity + 1) * multiplicity
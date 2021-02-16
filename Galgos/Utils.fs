module Utils 

let getMultipleSize multiplicity originalSize = 
    if originalSize % multiplicity = 0 then 
        originalSize
    else 
        (originalSize / multiplicity + 1) * multiplicity
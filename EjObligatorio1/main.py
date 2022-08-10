import pandas as pd

xls = pd.ExcelFile(r"dataset.xls") # use r before absolute file path 

sheetX = xls.parse(0) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis

var1 = sheetX['Sexo']

print(var1[1]) #1 is the row number...
import pandas as pd

df = pd.read_csv("ratings.csv")
columns = list(df.columns)

res = list(set(df["user_id"])) + list(set(df["chat_id"]))
res = {
    res[i]:i for i in range(len(res))    
}

result = {}
for i in columns:
    result[i] = []
    for j in df[i]: result[i].append(res[j]  if j in res else j)
    
print(result)
res = pd.DataFrame({
    i: result[i] for i in result
})
res.to_csv("final_ratings.csv", index=False)
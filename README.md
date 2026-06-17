# MyFlaskWithLangchain

這個專案主要是想以flask的方式提供一層最簡單的服務，使用langchain提供的方式做後端AI的串接與工具的調用，對外則開放兩個API  
-/predict  
  -一個get的API，做單次的應答  
/predict_image  
  -一個Post的API，接受圖片+文字的輸入，做單次的應答  
  
在CMD裡使用這樣的指令應該就能執行起來  
```bash  
python main.py   
```  

目前llmTools是空的，視需求可以自己加入更多工具  
目前系統提示詞是空的，視需求修改系統提示詞以獲得更好的回應  


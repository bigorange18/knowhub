'''
Author: bigorange chenorange2219@gmail.com
Date: 2025-01-04 10:22:20
LastEditors: royal-killer 1581279688@qq.com
LastEditTime: 2025-01-04 10:41:48
FilePath: \knowhub\gradio\base.py
Description: "

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import gradio 

# def start(name):
#     return f"hello {name}!"


# if __name__ == "__name__":
#     demo1 = gradio.Interface(
#         fn=start,
#         inputs='text',
#         outputs='text'
#     )
#     demo1.launch()


import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share=True)


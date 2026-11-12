import jieba


text = "我喜欢吃苹果，苹果很甜。"
cur_res = jieba.cut(text, cut_all=False)
print("Default Mode: " + "/ ".join(cur_res))  # 默认精确模式
print(f'Default Mode: {"".join(cur_res)}')
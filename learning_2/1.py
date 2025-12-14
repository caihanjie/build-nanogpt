import matplotlib.pyplot as plt

# 任务数据（按周）
tasks = [
    {"name": "项目启动",       "start": 1,  "duration": 2},
    {"name": "项目规划",       "start": 3,  "duration": 2},
    {"name": "需求阶段",       "start": 5,  "duration": 4},
    {"name": "设计阶段",       "start": 9,  "duration": 4},
    {"name": "开发阶段",       "start": 13, "duration": 8},
    {"name": "测试阶段",       "start": 21, "duration": 3},
    {"name": "部署与上线",     "start": 24, "duration": 1},
    {"name": "培训与推广",     "start": 17, "duration": 4},
    {"name": "项目收尾",       "start": 24, "duration": 1},
]

# 为了画图，准备 y 轴位置
task_names = [t["name"] for t in tasks]
y_pos = range(len(tasks))

fig, ax = plt.subplots(figsize=(10, 6))

for i, t in enumerate(tasks):
    # barh(left=开始周-1, width=持续周数)
    ax.barh(
        y=i,
        width=t["duration"],
        left=t["start"] - 1,   # 从0开始更直观
    )

# 设置 y 轴为任务名称
ax.set_yticks(list(y_pos))
ax.set_yticklabels(task_names, fontproperties='SimHei')  # SimHei 是中文字体名，如果报错可以去掉

# x 轴显示周数
ax.set_xlabel("周次")
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))  # 每2周一个刻度

# 标题
ax.set_title("G大学图书管理系统项目 甘特图", fontproperties='SimHei')

# 网格（可选）
ax.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

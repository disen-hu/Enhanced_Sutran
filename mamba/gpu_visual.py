import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter
import numpy as np
from datetime import datetime

class GPUMonitor:
    def __init__(self, update_interval=1, window_size=60):
        self.update_interval = update_interval  # 更新间隔(秒)
        self.window_size = window_size  # 显示窗口大小(秒)
        
        # 初始化数据存储
        self.timestamps = []
        self.gpu_utils = []
        self.memory_utils = []
        
        # 设置图表
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('GPU Panel', fontsize=16)
        
        # 设置子图
        self.setup_subplot(self.ax1, 'GPU Usage (%)', 0, 100)
        self.setup_subplot(self.ax2, 'GPU Memory Usage (%)', 0, 100)
        
        # 初始化线条
        self.line1, = self.ax1.plot([], [], 'g-', linewidth=2)
        self.line2, = self.ax2.plot([], [], 'b-', linewidth=2)

    def setup_subplot(self, ax, ylabel, ymin, ymax):
        ax.set_ylabel(ylabel)
        ax.set_xlabel('时间')
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
    
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', 
                 '--format=csv,nounits,noheader'
            ]).decode()
            gpu_util, mem_util = map(float, result.strip().split(','))
            return gpu_util, mem_util
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
            return 0, 0

    def update(self, frame):
        """更新图表数据"""
        # 获取当前GPU数据
        gpu_util, mem_util = self.get_gpu_info()
        current_time = datetime.now()
        
        # 更新数据列表
        self.timestamps.append(current_time)
        self.gpu_utils.append(gpu_util)
        self.memory_utils.append(mem_util)
        
        # 只保留窗口大小内的数据
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
            self.gpu_utils = self.gpu_utils[-self.window_size:]
            self.memory_utils = self.memory_utils[-self.window_size:]
        
        # 更新x轴范围
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(
                min(self.timestamps),
                max(self.timestamps)
            )
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        
        # 更新数据
        self.line1.set_data(self.timestamps, self.gpu_utils)
        self.line2.set_data(self.timestamps, self.memory_utils)
        
        # 添加当前值标签
        self.ax1.set_title(f'info GPU Usage: {gpu_util:.1f}%')
        self.ax2.set_title(f'Memory Usage: {mem_util:.1f}%')
        
        return self.line1, self.line2

    def start(self):
        """启动监控"""
        self.ani = FuncAnimation(
            self.fig, 
            self.update, 
            interval=self.update_interval * 1000,
            blit=True
        )
        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            print(f"显示错误: {e}")
        finally:
            plt.close('all')

if __name__ == "__main__":
    monitor = GPUMonitor(update_interval=1, window_size=60)
    monitor.start()
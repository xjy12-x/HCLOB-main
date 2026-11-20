# F:\contrast-11\run_hcsc.py
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from .main import main

    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"导入失败: {e}")
    print("请检查项目结构是否正确")

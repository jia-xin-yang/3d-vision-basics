# ex01_structure_scan.py
# 阶段 1 任务 1：MVTec 3D-AD 结构普查
# 目的：验证"完整性假设"——两笔对账：
#   对账 1: train + validation + test 总数 == 4147（官方总扫描数）
#   对账 2: 每类 gt 掩码数 == 该类 test 样本数（逐点分数能对上逐点掩码的前提）

from pathlib import Path
import csv

# ---- 配置区 ----
# 数据集不进仓库（13GiB 太大），脚本用绝对路径去读它
# 类别目录直接在根下：/mnt/e/datasets/mvtec3d/bagel/ ...
ROOT = Path('/mnt/e/datasets/mvtec3d')


def count_files(d: Path, pattern: str):
    """数目录 d 下匹配 pattern 的文件个数。
    目录不存在返回 None（用来报警结构异常），存在但没有匹配文件返回 0。"""
    if not d.is_dir():
        return None
    return len(list(d.glob(pattern)))


def scan_class(cls_dir: Path) -> dict:
    """普查一个类别目录（如 bagel/），返回该类的全部统计量。"""
    # train 和 validation 都只有 good，数 xyz/ 下的 tiff 即可
    train_good = count_files(cls_dir / 'train' / 'good' / 'xyz', '*.tiff')
    val_good = count_files(cls_dir / 'validation' / 'good' / 'xyz', '*.tiff')

    # test/ 下每个子目录是一种测试划分：good + 各缺陷类型
    test_dir = cls_dir / 'test'
    test_subs = sorted(p for p in test_dir.iterdir() if p.is_dir())

    test_good = None          # test/good 单独记（它是正常测试样本，不算缺陷）
    defects = {}              # 缺陷名 -> xyz tiff 数
    gt_counts = {}            # test 下每个子目录（含 good）的 gt png 数

    for sub in test_subs:
        name = sub.name
        xyz_n = count_files(sub / 'xyz', '*.tiff')
        gt_n = count_files(sub / 'gt', '*.png')
        gt_counts[name] = gt_n
        if name == 'good':
            test_good = xyz_n
        else:
            defects[name] = xyz_n

    defect_total = sum(defects.values())
    test_total = test_good + defect_total   # 该类全部测试样本
    gt_total = sum(gt_counts.values())      # 该类全部掩码（每个 test 样本一张，含 good）

    return {
        'class': cls_dir.name,
        'train': train_good,
        'val': val_good,
        'test_good': test_good,
        'defects': defects,                # dict，打印时展开看
        'n_defect_types': len(defects),
        'defect_total': defect_total,
        'test_total': test_total,
        'gt_total': gt_total,
        'aligned': gt_total == test_total, # 逐类对账结果
    }


def main():
    # ---- 第 1 步：语义过滤，找出真正的类别目录 ----
    # 根目录下除了 10 个类别还有一个 MVTEC_3D-AD/（下载残留，里面只有压缩包）
    # 判断标准：同时含有 train 和 test 子目录 → 是类别目录
    # （用结构特征判断，比写死 10 个名字健壮，换数据集也能用）
    categories = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir():
            continue
        subs = {q.name for q in p.iterdir() if q.is_dir()}
        if 'train' in subs and 'test' in subs:
            categories.append(p)

    print(f'发现 {len(categories)} 个类别: {[p.name for p in categories]}')
    # 验证不过就停：类别数不对就当场炸，不往下跑
    assert len(categories) == 10, '类别数不是 10！先检查过滤条件或数据是否完好'

    # ---- 第 2/3 步：逐类普查，累积成表 ----
    rows = [scan_class(p) for p in categories]

    header = (f"{'类别':<14}{'train':>6}{'val':>5}{'test_good':>10}"
              f"{'test_缺陷(类型数)':>18}{'test合计':>9}{'gt':>6}  对齐")
    print(header)
    print('-' * (len(header) + 4))

    for r in rows:
        defect_str = f"{r['defect_total']}({r['n_defect_types']})"
        mark = '✓' if r['aligned'] else '✗'
        print(f"{r['class']:<14}{r['train']:>6}{r['val']:>5}{r['test_good']:>10}"
              f"{defect_str:>18}{r['test_total']:>9}{r['gt_total']:>6}  {mark}")
        # 逐类展开缺陷明细，方便第 2 步人工核对
        for name, n in r['defects'].items():
            print(f"{'':<14}└ {name}: {n}")

    # ---- 第 4 步：合计 + 两笔对账 ----
    tot_train = sum(r['train'] for r in rows)
    tot_val = sum(r['val'] for r in rows)
    tot_test_good = sum(r['test_good'] for r in rows)
    tot_defect = sum(r['defect_total'] for r in rows)
    tot_test = sum(r['test_total'] for r in rows)
    tot_gt = sum(r['gt_total'] for r in rows)
    grand = tot_train + tot_val + tot_test

    print('-' * (len(header) + 4))
    print(f"{'合计':<14}{tot_train:>6}{tot_val:>5}{tot_test_good:>10}"
          f"{tot_defect:>18}{tot_test:>9}{tot_gt:>6}")

    print()
    ok1 = '✓' if grand == 4147 else '✗'
    print(f'对账 1: train+val+test = {tot_train} + {tot_val} + {tot_test} = {grand}  '
          f'(目标 4147)  {ok1}')
    ok2 = '✓' if tot_gt == tot_test else '✗'
    print(f'对账 2: gt 合计 = {tot_gt}  vs  test 合计 = {tot_test}  {ok2}')

    # 加分观察：rgb png 与样本一一对应，总数应该也是 4147
    rgb_total = len(list(ROOT.rglob('rgb/*.png')))
    ok3 = '✓' if rgb_total == grand else '✗'
    print(f'观察  : rgb png 总数 = {rgb_total}  (应等于样本总数 {grand})  {ok3}')

    # ---- 第 5 步：落盘 CSV（本阶段第一个中间产物，任务 5 还要引用它）----
    csv_path = Path('ex01_structure_scan.csv')
    # utf-8-sig: 带 BOM 的 utf-8，Windows 的 Excel 直接双击打开中文不乱码
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'train_good', 'val_good', 'test_good',
                    'defect_types', 'defect_total', 'test_total', 'gt_total', 'gt_aligned'])
        for r in rows:
            w.writerow([r['class'], r['train'], r['val'], r['test_good'],
                        r['n_defect_types'], r['defect_total'],
                        r['test_total'], r['gt_total'], r['aligned']])
        w.writerow(['TOTAL', tot_train, tot_val, tot_test_good,
                    '', tot_defect, tot_test, tot_gt, tot_gt == tot_test])
    print(f'\nCSV 已写入: {csv_path.resolve()}')


if __name__ == '__main__':
    main()

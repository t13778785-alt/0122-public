#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from types import MethodType

# 配置loguru日志
from loguru import logger

import shutil
import subprocess

# 增强shutil.rmtree删除能力，解决s3挂载目录删除失败问题
# - 必须在导入 lerobot 之前
_original_rmtree = shutil.rmtree

def safe_rmtree(path, ignore_errors=False, onerror=None):
    try:
        _original_rmtree(path, ignore_errors, onerror)
    except OSError as e:
        if e.errno == 39:  # Directory not empty
            subprocess.run(['rm', '-rf', str(path)], check=False)
        elif not ignore_errors:
            raise

shutil.rmtree = safe_rmtree


# 设置日志文件路径
log_file = Path(os.environ.get('OUTPUT_PATH', '/tmp')) / 'export.log'

# 移除默认的日志处理器并添加自定义的文件和控制台输出
logger.remove()
logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss,SSS} - {level} - {message}", level="INFO", rotation="100 MB")
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss,SSS} - {level} - {message}", level="INFO")


def main():
    namespace = os.environ.get("NAMESPACE", os.environ.get("HBASE_NAMESPACE", ""))
    table = os.environ.get("TABLE", os.environ.get("HBASE_TABLE", ""))
    episode_ids_raw = os.environ.get("EPISODE_IDS", "")
    episode_ids = [s for s in (episode_ids_raw.split(",") if episode_ids_raw else [])] or None
    # 添加 PARSE_RECORDS 参数支持
    parse_records_raw = os.environ.get("PARSE_RECORDS", "")
    parse_records = [s for s in (parse_records_raw.split(",") if parse_records_raw else [])] or None
    # 添加 ANNOTATION_TASK_ID 参数支持
    annotation_task_id = os.environ.get("ANNOTATION_TASK_ID")
    output_root = os.environ.get("OUTPUT_PATH", "/data/out")
    repo_id = os.environ.get("DATASET_REPO_ID", os.environ.get("REPO_ID", "my_dataset/my_pusht"))

    # 日志配置已提前设置，此处不再重复配置

    # import query and exporter classes
    from dp_lerobot_parser.database.hbase_episode_frame_query import HBaseEpisodeFrameQueryWrapper
    from dp_lerobot_parser.exporter.dp_lerobot_exporter import DpLerobotExportConfig, DpLerobotExporter
    from dp_lerobot_parser.exporter.feature.feature_exporters import (
        EpisodeStepActionExporter,
        EpisodeStepRewardExporter,
        EpisodeStepTruncatedExporter,
        CameraExporter,
        EpisodeStepTerminatedExporter,
        EpisodeStepObservationExporter,
    )

    # 如果提供了 PARSE_RECORDS，则使用它来查询数据
    if parse_records:
        # 我们需要自定义查询逻辑来支持基于 parse_record_id 前缀的查询
        # 通过继承类来添加自定义的query_episodes方法
        class CustomHBaseEpisodeFrameQueryWrapper(HBaseEpisodeFrameQueryWrapper):
            def query_episodes(self):
                return self.query_episodes_by_prefix(parse_records)
        
        query = CustomHBaseEpisodeFrameQueryWrapper(
            namespace=namespace, 
            table=table,
            query={"annotation_task_id": annotation_task_id} if annotation_task_id else None
        )
    else:
        query = HBaseEpisodeFrameQueryWrapper(
            namespace=namespace, 
            table=table, 
            episode_ids=episode_ids,
            query={"annotation_task_id": annotation_task_id} if annotation_task_id else None
        )

    exporters = [
        # EpisodeStepActionExporter(shape=(2,), names={"motors": ["motor_0", "motor_1"]}),
        # EpisodeStepObservationExporter(shape=(2,), names={"motors": ["motor_0", "motor_1"]}),
        CameraExporter(key="cam_h.color.image_raw.compressed", shape=(480, 848, 3), names=["height", "width", "channel"]),
        # CameraExporter(key="cam_r.color.image_raw.compressed", shape=(480, 848, 3), names=["height", "width", "channel"]),
        # EpisodeStepTruncatedExporter(),
        # EpisodeStepRewardExporter(),
        # EpisodeStepTerminatedExporter(),
    ]

    config = DpLerobotExportConfig(
        dataset_repo_id=repo_id,
        dataset_root=Path(output_root),
        fps=int(os.environ.get("FPS", "10")),
        tolerance_s=0.5,
        exporters=exporters,
    )

    exporter = DpLerobotExporter(query=query, config=config)
    try:
        exporter.export()
        logger.info("导出成功完成!")
        logger.info(f"数据已导出到目录: {os.path.join(output_root, repo_id)}")
    except Exception as e:
        logger.error(f"导出过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()
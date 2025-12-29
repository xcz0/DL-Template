#!/bin/bash

# 下载模型权重和 tensorboard 日志文件
set -e

# 检查 .env 文件并加载环境变量
if [ -f .env ]; then
	set -a
	source .env
	set +a
fi

# 检查必要的环境变量
if [ -z "$CHECKPOINT_PATH" ]; then
	echo "请在 .env 文件中设置 CHECKPOINT_PATH 环境变量"
	exit 1
fi

BASE_URL="https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
PRETRAINED_FILES=( \
	"GoogleNet.ckpt" \
	"ResNet.ckpt" \
	"ResNetPreAct.ckpt" \
	"DenseNet.ckpt" \
	"tensorboards/GoogleNet/events.out.tfevents.googlenet" \
	"tensorboards/ResNet/events.out.tfevents.resnet" \
	"tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact" \
	"tensorboards/DenseNet/events.out.tfevents.densenet" \
)

# 创建 checkpoint 目录
mkdir -p "$CHECKPOINT_PATH"

for file_name in "${PRETRAINED_FILES[@]}"; do
	file_path="$CHECKPOINT_PATH/$file_name"
	dir_path="$(dirname "$file_path")"
	mkdir -p "$dir_path"
	if [ ! -f "$file_path" ]; then
		file_url="$BASE_URL$file_name"
		echo "Downloading $file_url ..."
		if ! curl -fSL "$file_url" -o "$file_path"; then
			echo "下载 $file_name 失败，请手动下载或联系作者。"
		fi
	else
		echo "$file_name 已存在，跳过下载。"
	fi
done

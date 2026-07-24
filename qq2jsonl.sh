#!/bin/bash

# 1. 检查是否传入了参数 (name)
if [ -z "$1" ]; then
    echo "错误: 未提供文件前缀名称。"
    echo "用法: $0 <name>"
    echo "示例: $0 messages (将会把 messages.json 转换为 messages.jsonl)"
    exit 1
fi

NAME="$1"
INPUT_FILE="${NAME}.json"
OUTPUT_FILE="${NAME}.jsonl"

# 2. 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 '$INPUT_FILE'。"
    exit 1
fi

echo "正在处理 ${INPUT_FILE} -> ${OUTPUT_FILE} ..."

# 3. 执行 jq 一步到位提取并过滤
# 逻辑：展开 messages 数组 -> 根据 content.text 过滤无用信息 -> 重组为单层 JSON 对象
jq -c '
  .messages[] 
  | select(
      (.content.text | test("^\\[(图片|图片:.*|\\d+|表情|视频)\\]$") | not) 
      and .content.text != ""
    ) 
  | {
      time: .timestamp, 
      uid: .sender.uin, 
      name: .sender.name, 
      text: .content.text
    }
' "$INPUT_FILE" > "$OUTPUT_FILE"

# 4. 检查是否执行成功
if [ $? -eq 0 ]; then
    echo "转换成功！结果已保存至: ${OUTPUT_FILE}"
else
    echo "转换失败，请检查 jq 命令或 JSON 文件格式。"
    exit 1
fi
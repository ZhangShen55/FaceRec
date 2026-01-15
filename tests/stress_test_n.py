"""
人脸识别 API 压力测试脚本 - 1000 并发
测试 /recognize 接口在高并发下的性能表现
"""
import asyncio
import aiohttp
import mimetypes
import base64
import time
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter
import json

# 测试配置
API_URL = "http://localhost:8004/recognize"
IMAGE_PATH = "/root/workspace/FaceRecAPI_DEV/app/tests/常泽宇.png"
CONCURRENT_REQUESTS = 100  # 并发数
THRESHOLD = 0.25  # 识别阈值

# 统计数据
results = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "errors": [],
    "response_times": [],
    "status_codes": Counter(),
    "matched_persons": Counter()
}


def load_image_base64() -> str:
    """加载测试图片并转换为 base64"""
    # if not IMAGE_PATH.exists():
    #     raise FileNotFoundError(f"测试图片不存在: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        print(f"base64长度: {len(base64_str)}")
        return base64_str

def load_image_base64_with_header(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/jpeg"  # 给个默认值也行

    return f"data:{mime};base64,{b64}"


async def send_request(session: aiohttp.ClientSession, request_id: int, photo_base64: str) -> Dict:
    """发送单个识别请求"""
    start_time = time.time()

    try:
        payload = {
            "photo": photo_base64,
            "threshold": THRESHOLD
        }

        async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=400)) as response:
            response_time = time.time() - start_time
            status_code = response.status

            try:
                result = await response.json()
            except Exception as e:
                result = {"error": f"JSON解析失败: {str(e)}"}

            return {
                "request_id": request_id,
                "status_code": status_code,
                "response_time": response_time,
                "result": result,
                "success": status_code == 200
            }

    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time": response_time,
            "result": {"error": "请求超时"},
            "success": False
        }

    except Exception as e:
        response_time = time.time() - start_time
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time": response_time,
            "result": {"error": str(e)},
            "success": False
        }


async def run_concurrent_test(photo_base64: str, concurrent: int):
    """运行并发测试"""
    print(f"\n{'='*70}")
    print(f"🚀 开始压力测试")
    print(f"{'='*70}")
    print(f"API 地址: {API_URL}")
    print(f"并发数: {concurrent}")
    print(f"阈值: {THRESHOLD}")
    print(f"测试图片: {IMAGE_PATH}")
    print(f"{'='*70}\n")

    # 创建连接器（增加连接池大小）
    connector = aiohttp.TCPConnector(limit=concurrent, limit_per_host=concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建所有任务
        tasks = [
            send_request(session, i, photo_base64)
            for i in range(concurrent)
        ]

        print(f"⏳ 正在发送 {concurrent} 个并发请求...")
        start_time = time.time()

        # 并发执行所有请求
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # 统计结果
        for response in responses:
            if isinstance(response, Exception):
                results["failed"] += 1
                results["errors"].append(str(response))
            else:
                results["total"] += 1
                results["response_times"].append(response["response_time"])
                results["status_codes"][response["status_code"]] += 1

                if response["success"]:
                    results["success"] += 1

                    # 统计识别结果
                    result_data = response["result"]
                    if isinstance(result_data, dict):
                        status_code = result_data.get("statusCode", 0)
                        data = result_data.get("data", {})

                        if status_code == 200 and data:
                            # 识别成功
                            matches = data.get("match", [])
                            if matches:
                                # 记录第一个匹配的人员
                                first_match = matches[0]
                                person_key = f"{first_match.get('name')}_{first_match.get('number')}"
                                results["matched_persons"][person_key] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(response["result"].get("error", "未知错误"))

        return total_time


def print_statistics(total_time: float):
    """打印统计信息"""
    print(f"\n{'='*70}")
    print(f"📊 测试结果统计")
    print(f"{'='*70}\n")

    # 基础统计
    print(f"📈 基础指标:")
    print(f"  总请求数: {results['total']}")
    print(f"  成功请求: {results['success']} ({results['success']/results['total']*100:.2f}%)")
    print(f"  失败请求: {results['failed']} ({results['failed']/results['total']*100:.2f}%)")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  QPS (每秒请求数): {results['total']/total_time:.2f}")

    # 响应时间统计
    if results["response_times"]:
        response_times = sorted(results["response_times"])
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        p50 = response_times[int(len(response_times) * 0.5)]
        p90 = response_times[int(len(response_times) * 0.9)]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]

        print(f"\n⏱️  响应时间 (秒):")
        print(f"  最小值: {min_time:.3f}s")
        print(f"  最大值: {max_time:.3f}s")
        print(f"  平均值: {avg_time:.3f}s")
        print(f"  P50: {p50:.3f}s")
        print(f"  P90: {p90:.3f}s")
        print(f"  P95: {p95:.3f}s")
        print(f"  P99: {p99:.3f}s")

    # HTTP 状态码分布
    print(f"\n📡 HTTP 状态码分布:")
    for status_code, count in results["status_codes"].most_common():
        print(f"  {status_code}: {count} ({count/results['total']*100:.2f}%)")

    # 识别结果统计
    if results["matched_persons"]:
        print(f"\n👤 识别结果 (匹配人员):")
        for person, count in results["matched_persons"].most_common(10):
            print(f"  {person}: {count} 次 ({count/results['success']*100:.2f}%)")

    # 错误统计
    if results["errors"]:
        print(f"\n❌ 错误统计 (前 10 个):")
        error_counter = Counter(results["errors"])
        for error, count in error_counter.most_common(10):
            print(f"  {error}: {count} 次")

    print(f"\n{'='*70}\n")


def export_results(total_time: float):
    """导出测试结果到 JSON 文件"""
    output = {
        "test_config": {
            "api_url": API_URL,
            "concurrent_requests": CONCURRENT_REQUESTS,
            "threshold": THRESHOLD,
            "image_path": str(IMAGE_PATH)
        },
        "summary": {
            "total_requests": results["total"],
            "success_requests": results["success"],
            "failed_requests": results["failed"],
            "total_time": round(total_time, 2),
            "qps": round(results["total"] / total_time, 2)
        },
        "response_times": {
            "min": round(min(results["response_times"]), 3) if results["response_times"] else 0,
            "max": round(max(results["response_times"]), 3) if results["response_times"] else 0,
            "avg": round(sum(results["response_times"]) / len(results["response_times"]), 3) if results["response_times"] else 0,
        },
        "status_codes": dict(results["status_codes"]),
        "matched_persons": dict(results["matched_persons"])
    }

    # output_file = Path(__file__).parent / f"stress_test_result_{int(time.time())}.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(output, f, ensure_ascii=False, indent=2)

    # print(f"📄 测试结果已导出到: {output_file}")


async def main():
    """主函数"""
    try:
        # 加载测试图片
        print("📸 加载测试图片...")
        photo_base64 = load_image_base64_with_header(IMAGE_PATH)
        print(f"✅ 测试图片加载成功 (大小: {len(photo_base64)} 字符)")

        # 运行并发测试
        total_time = await run_concurrent_test(photo_base64, CONCURRENT_REQUESTS)

        # 打印统计信息
        print_statistics(total_time)

        # 导出结果
        export_results(total_time)

        # 性能评估
        qps = results["total"] / total_time
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0
        success_rate = results["success"] / results["total"] * 100 if results["total"] > 0 else 0

        print(f"🎯 性能评估:")
        if success_rate >= 99 and qps >= 50 and avg_time <= 0.5:
            print(f"  ✅ 优秀！系统在高并发下表现出色")
        elif success_rate >= 95 and qps >= 30:
            print(f"  ✅ 良好！系统性能符合预期")
        elif success_rate >= 90:
            print(f"  ⚠️  一般，建议优化系统性能")
        else:
            print(f"  ❌ 较差，系统需要优化")

        print(f"\n💡 优化建议:")
        if avg_time > 0.5:
            print(f"  - 平均响应时间较慢，可考虑:")
            print(f"    • 检查 Redis 缓存是否生效")
            print(f"    • 增加 uvicorn workers 数量")
            print(f"    • 优化 AI 模型推理速度")

        if success_rate < 99:
            print(f"  - 成功率低于 99%，可考虑:")
            print(f"    • 增加服务器资源 (CPU/内存)")
            print(f"    • 调整连接池大小")
            print(f"    • 检查网络带宽")

        if qps < 50:
            print(f"  - QPS 较低，可考虑:")
            print(f"    • 启用 Redis 缓存加速")
            print(f"    • 使用多进程部署 (uvicorn workers)")
            print(f"    • 考虑使用 GPU 加速")

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠️  测试被用户中断")
        sys.exit(0)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 设置事件循环策略（Windows 兼容）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行测试
    asyncio.run(main())

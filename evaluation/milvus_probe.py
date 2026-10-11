"""只读检查 Milvus 的 HTTPS/gRPC 通路，不输出凭证。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    parser.add_argument("--grpc", action="store_true")
    parser.add_argument("--collection")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri = os.environ["MILVUS_URI"].rstrip("/")
    if args.grpc:
        if args.proxy:
            os.environ["grpc_proxy"] = args.proxy
        from pymilvus import MilvusClient
        client = None
        try:
            client = MilvusClient(uri=uri, token=os.environ["MILVUS_TOKEN"], timeout=10)
            names = client.list_collections(timeout=10)
            print(json.dumps({"transport": "grpc", "collections": names}, ensure_ascii=False))
        except Exception as error:
            print(json.dumps({"transport": "grpc", "error_type": type(error).__name__,
                              "success": False}, ensure_ascii=False))
            raise SystemExit(1) from None
        finally:
            if client is not None:
                client.close()
    else:
        try:
            with httpx.Client(proxy=args.proxy, timeout=15) as client:
                if not args.count_only:
                    response = client.post(uri + "/v2/vectordb/collections/list",
                                           headers={"Authorization": "Bearer " + os.environ["MILVUS_TOKEN"]},
                                           json={})
                    try:
                        payload = response.json()
                        print(json.dumps({"transport": "https", "http_status": response.status_code,
                                          "code": payload.get("code"), "collections": payload.get("data")},
                                         ensure_ascii=False))
                    except ValueError:
                        print(json.dumps({"transport": "https", "http_status": response.status_code}))
                if args.collection:
                    for operation, body in (
                        ("collections/describe", {"collectionName": args.collection}),
                        ("entities/query", {"collectionName": args.collection,
                                            "filter": "", "outputFields": ["count(*)"],
                                            "consistencyLevel": "Strong"}),
                    ):
                        if args.count_only and operation == "collections/describe":
                            continue
                        response = client.post(uri + "/v2/vectordb/" + operation,
                                               headers={"Authorization": "Bearer " + os.environ["MILVUS_TOKEN"]},
                                               json=body)
                        print(operation, json.dumps(response.json(), ensure_ascii=False))
        except httpx.HTTPError as error:
            print(json.dumps({"transport": "https", "error_type": type(error).__name__, "success": False}))
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# config.yml 경로
CONFIG_PATH = Path(__file__).parent / "config.yml"

# YAML 파일 로드
with open(CONFIG_PATH, "r") as f:
    yaml_config = yaml.safe_load(f)


@dataclass
class APIConfig:
    """API 설정 (YAML 기반)"""
    openai_api_key: str = yaml_config["openai"]["api_key"]
    openai_org_id: Optional[str] = yaml_config["openai"].get("org_id")
    request_timeout: int = yaml_config["openai"].get("request_timeout", 30)
    max_retries: int = yaml_config["openai"].get("max_retries", 3)
    retry_delay: float = yaml_config["openai"].get("retry_delay", 1.0)


# 전역 인스턴스
api_config = APIConfig()


# config.yml 파일에 아래와 같이 작성해서 사용하시면 됩니다..!
# openai:
#   api_key: "sk-여기에_API_KEY_입력"
#   org_id: null
#   request_timeout: 30
#   max_retries: 3
#   retry_delay: 1.0
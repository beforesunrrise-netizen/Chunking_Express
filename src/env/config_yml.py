import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    """API 설정 (환경변수 기반)"""

    openai_org_id: Optional[str] = os.getenv("OPENAI_ORG_ID")
    request_timeout: int = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("OPENAI_RETRY_DELAY", "1.0"))


# 전역 인스턴스
api_config = APIConfig()


# config.yml 파일에 아래와 같이 작성해서 사용하시면 됩니다..!
# 위치 env.config.yml
# openai:
#   api_key: "sk-여기에_API_KEY_입력"
#   org_id: null
#   request_timeout: 30
#   max_retries: 3
#   retry_delay: 1.0
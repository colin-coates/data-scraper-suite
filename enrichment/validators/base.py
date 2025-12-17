from typing import Dict, Any

class BaseValidator:
    def validate(self, data: Dict[str, Any]) -> bool:
        raise NotImplementedError

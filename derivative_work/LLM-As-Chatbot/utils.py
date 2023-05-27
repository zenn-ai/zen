
from chats import vicuna

from pingpong.gradio import GradioVicunaChatPPManager

from pingpong.pingpong import PPManager
from pingpong.pingpong import PromptFmt
from pingpong.pingpong import UIFmt
from pingpong.gradio import GradioChatUIFmt

class BaizePromptFmt(PromptFmt):
    @classmethod
    def ctx(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
"""
        
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""[|Human|]: {ping}
[|AI|]: {pong}
"""

class BaizeChatPPManager(PPManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1, fmt: PromptFmt=BaizePromptFmt, truncate_size: int=None):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
            
        results = fmt.ctx(self.ctx)
        
        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += fmt.prompt(pingpong, truncate_size=truncate_size)
            
        return results

class GradioBaizeChatPPManager(BaizeChatPPManager):
    def build_uis(self, from_idx: int=0, to_idx: int=-1, fmt: UIFmt=GradioChatUIFmt):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = []
        
        for pingpong in self.pingpongs[from_idx:to_idx]:
            results.append(fmt.ui(pingpong))
            
        return results    
    
def get_chat_interface(model_type):
    return vicuna.chat_stream

def get_chat_manager(model_type):
    return GradioVicunaChatPPManager()
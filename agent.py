import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np

from model_wrapper import query_vlm
from sandbox import execute as sandbox_execute
from utils import extract_answer, extract_code

@dataclass
class StepRecord:
    k: int
    plan: str = ""
    code: Optional[str] = None
    execution: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"k": self.k, "plan": self.plan}
        if self.code: d["code"] = self.code
        if self.execution: d["execution"] = self.execution
        if self.answer: d["answer"] = self.answer
        return d


@dataclass
class AgentResult:
    prediction: str
    steps: List[StepRecord] = field(default_factory=list)
    method: str = "agent"

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "method": self.method,
            "steps": [s.to_dict() for s in self.steps],
        }

# ───────────────────────── task type detection ─────────────────────────

def _detect_task_type(question: str) -> str:
    q = question.lower()
    if "how many" in q or "count" in q: return "count_shapes"
    if "tallest bar" in q or "highest value" in q or "value of the" in q: return "bar_chart_max"
    if "angle" in q and "rectangle" not in q: return "line_angle"
    if "percentage" in q or "covered" in q or "fraction" in q: return "region_fraction"
    if "most" in q and ("color" in q or "colour" in q): return "most_common_color"
    return "unknown"

def _parse_color_and_shape(question: str):
    import re
    q = question.lower()
    colors = ["red", "green", "blue", "purple", "orange", "yellow", "black"]
    shapes = ["circle", "square", "triangle", "rectangle"]
    found_color = next((c for c in colors if re.search(r'\b' + c + r'\b', q)), None)
    found_shape = next((s for s in shapes if re.search(r'\b' + s + r'\b', q)), None)
    return found_color, found_shape

_HSV_RANGES = {
    "red":    "mask = cv2.inRange(hsv, np.array([0,20,20]), np.array([15,255,255])) | cv2.inRange(hsv, np.array([160,20,20]), np.array([180,255,255]))",
    "green":  "mask = cv2.inRange(hsv, np.array([30,15,15]), np.array([95,255,255]))",
    "blue":   "mask = cv2.inRange(hsv, np.array([90,20,20]), np.array([150,255,255]))",
    "purple": "mask = cv2.inRange(hsv, np.array([125,5,5]), np.array([175,255,255]))",
}

_SHAPE_CHECK = {
    "circle":   "circ = (4*math.pi*area)/(peri*peri) if peri>0 else 0\n        if circ > 0.6 and area > 10: count += 1",
    "square":   "approx = cv2.approxPolyDP(cnt, 0.04*peri, True)\n        if len(approx)==4 and area>10:\n            x_,y_,w_,h_ = cv2.boundingRect(approx)\n            asp = float(w_)/h_ if h_>0 else 0\n            if 0.5 < asp < 1.5: count += 1",
    "triangle": "approx = cv2.approxPolyDP(cnt, 0.04*peri, True)\n        if len(approx)==3 and area>10: count += 1",
    "rectangle":"approx = cv2.approxPolyDP(cnt, 0.04*peri, True)\n        if len(approx)==4 and area>10: count += 1",
}


def _build_count_code(color, shape):
    mask_code = _HSV_RANGES.get(color, _HSV_RANGES["blue"])
    shape_code = _SHAPE_CHECK.get(shape, _SHAPE_CHECK["circle"])
    return f"img=np.array(image); hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)\n{mask_code}\ncontours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\ncount=0\nfor cnt in contours:\n    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)\n    {shape_code}\nresult['answer']=str(count)"

def _build_region_fraction_code(color):
    mask_code = _HSV_RANGES.get(color, _HSV_RANGES["green"])
    # Use the full image area for "percentage of the image" tasks
    return f"img=np.array(image); hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)\n{mask_code}\nentire_area = img.shape[0]*img.shape[1]\nperc=int(round(100.0*cv2.countNonZero(mask)/entire_area))\nresult['answer']=str(perc)"

def _build_most_common_color_code(shape):
    shape_snippet = _SHAPE_CHECK.get(shape, _SHAPE_CHECK["circle"]).replace("count += 1", "temp = 1")
    return f"from collections import Counter; img=np.array(image); hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)\ngray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)\n_,bin=cv2.threshold(gray,230,255,cv2.THRESH_BINARY_INV)\ncontours,_=cv2.findContours(bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\ncounts=Counter()\nfor cnt in contours:\n    area=cv2.contourArea(cnt); peri=cv2.arcLength(cnt,True)\n    if area<10: continue\n    temp=0\n    {shape_snippet}\n    if temp==0: continue\n    m=np.zeros(gray.shape,np.uint8); cv2.drawContours(m,[cnt],-1,255,-1)\n    h=cv2.mean(hsv[:,:,0],mask=m)[0]\n    if (h<20 or h>160): c='red'\n    elif 30<h<95: c='green'\n    elif 90<h<150: c='blue'\n    elif 125<h<175: c='purple'\n    else: c=None\n    if c: counts[c]+=1\nresult['answer']=counts.most_common(1)[0][0] if counts else 'red'"

def _build_line_angle_code():
    return '''img=np.array(image); gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)\n_,bin=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)\nlines=cv2.HoughLinesP(bin,1,np.pi/180,threshold=30,minLineLength=30,maxLineGap=20)\nif lines is not None and len(lines)>=2:\n    angles=[]\n    for line in lines:\n        x1,y1,x2,y2 = line[0]; f = math.atan2(y2-y1, x2-x1)\n        if not angles:\n            angles.append(f)\n        else:\n            if all(abs(f - ex) > 0.3 and abs(abs(f - ex) - math.pi) > 0.3 for ex in angles):\n                angles.append(f)\n                if len(angles) == 2: break\n    if len(angles) >= 2:\n        deg = abs(math.degrees(angles[0]-angles[1]))\n        if deg > 90: deg = 180 - deg\n        result['answer'] = str(int(round(deg)))\n    else: result['answer'] = '0'\nelse: result['answer'] = '45\''''

def _build_bar_max_code():
    return "img=np.array(image); hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)\nmask=cv2.inRange(hsv,np.array([90,20,20]),np.array([140,255,255]))\ncontours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\nif not contours:\n    mask=cv2.inRange(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),0,200)\n    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)\nbars=[cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[3]>10]\nif not bars: result['answer']='0'\nelse:\n    bars.sort(key=lambda b: b[0]); bottom=max(b[1]+b[3] for b in bars)\n    tallest_h=max(bottom-b[1] for b in bars)\n    result['answer']=str(int(round((tallest_h/(img.shape[0]*0.75))*10)))"


def solve(model, image: Image.Image, question: str, max_steps: int = 3, timeout: float = 5.0) -> AgentResult:
    steps: List[StepRecord] = []
    task = _detect_task_type(question)
    color, shape = _parse_color_and_shape(question)

    # ── Step 1: Tool Execution ──
    code = None
    if task == "count_shapes" and color and shape: code=_build_count_code(color, shape)
    elif task == "bar_chart_max": code=_build_bar_max_code()
    elif task == "line_angle": code=_build_line_angle_code()
    elif task == "region_fraction" and color: code=_build_region_fraction_code(color)
    elif task == "most_common_color" and shape: code=_build_most_common_color_code(shape)
    
    if code:
        res = sandbox_execute(code, image, timeout=timeout)
        ans = str(res.result.get("answer")) if res.success and res.result.get("answer") is not None else None
        steps.append(StepRecord(k=0, plan=f"Precision tool: {task}", code=code, execution=res.__dict__, answer=ans))
        
        is_bad_angle = (task == "line_angle" and ans == "45")
        if res.success and ans is not None and not is_bad_angle:
            return AgentResult(prediction=ans, steps=steps, method="agent_code")

    # ── Step 2: Fallback to VLM ──
    v_ans = query_vlm(model, image, question)
    steps.append(StepRecord(k=len(steps), plan="Fallback VQA", answer=extract_answer(v_ans)))
    return AgentResult(prediction=extract_answer(v_ans), steps=steps, method="fallback_vqa")

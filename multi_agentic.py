import genesis as gs
import numpy as np
import requests
import base64
import io
import re
import threading
from PIL import Image
import time

# --- エージェント間共有データ (Blackboard) ---
class Blackboard:
    def __init__(self):
        self.target_pos = np.array([0.5, 0.2, 0.03])
        self.is_thinking = False

# --- 1. Perception Agent (認識) ---
class PerceptionAgent:
    def __init__(self, cameras):
        self.cameras = cameras

    def _encode(self, img_np):
        if isinstance(img_np, tuple): img_np = img_np[0]
        img = Image.fromarray(img_np)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def run_inference(self, blackboard):
        """Ollamaを使用してターゲット座標を更新し続ける"""
        while True:
            try:
                blackboard.is_thinking = True
                imgs = [self._encode(cam.render()[0]) for cam in self.cameras]
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llava",
                        "prompt": "Focus on Image 3 (Hand camera). Is the red box centered? Return precise [x, y, z].",
                        "images": imgs, "stream": False
                    },
                    timeout=8
                )
                text = response.json().get('response', "")
                match = re.search(r"\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]", text)
                if match:
                    new_coords = np.array([float(match.group(i)) for i in range(1, 4)])
                    # アルゴリズム維持: 80%反映
                    blackboard.target_pos = 0.2 * blackboard.target_pos + 0.8 * new_coords
                    print(f"[Perception Agent] Update: {blackboard.target_pos}")
            except:
                pass
            finally:
                blackboard.is_thinking = False
            time.sleep(0.01)

# --- 2. Brain Agent (意志決定・状態管理) ---
class BrainAgent:
    def __init__(self):
        self.state = "APPROACH"
        self.step_count = 0
        self.state_start_step = 0

    def update_logic(self, ee_pos, target_pos):
        """状態遷移アルゴリズムを司る"""
        elapsed = self.step_count - self.state_start_step
        next_state = self.state
        start_new_step = False

        if self.state == "APPROACH":
            dist = np.linalg.norm(np.array([target_pos[0], target_pos[1], 0.25]) - ee_pos)
            if dist < 0.04 or elapsed > 1500:
                next_state, start_new_step = "DESCEND", True

        elif self.state == "DESCEND":
            if ee_pos[2] < 0.055 or elapsed > 3000:
                next_state, start_new_step = "GRASP", True

        elif self.state == "GRASP":
            if elapsed > 200:
                next_state, start_new_step = "LIFT", True

        if start_new_step:
            self.state = next_state
            self.state_start_step = self.step_count
        
        self.step_count += 1
        return self.state

# --- 3. Controller Agent (物理制御) ---
class ControllerAgent:
    def __init__(self, robot, ee_link):
        self.robot = robot
        self.ee_link = ee_link
        self.smooth_goal_xyz = np.array([0.5, 0.0, 0.5])

    def move(self, state, target_pos):
        """状態に基づいた物理移動の実行"""
        # 目標座標と幅の決定
        if state == "APPROACH":
            final_target, gripper_width = np.array([target_pos[0], target_pos[1], 0.25]), 0.04
        elif state == "DESCEND":
            final_target, gripper_width = np.array([target_pos[0], target_pos[1], 0.025]), 0.04
        elif state == "GRASP":
            final_target, gripper_width = np.array([target_pos[0], target_pos[1], 0.025]), 0.0
        elif state == "LIFT":
            final_target, gripper_width = np.array([target_pos[0], target_pos[1], 0.4]), 0.0
        else:
            final_target, gripper_width = target_pos, 0.04

        # LERP & IK
        lerp_speed = 0.005 if state == "DESCEND" else 0.02
        self.smooth_goal_xyz = (1 - lerp_speed) * self.smooth_goal_xyz + lerp_speed * final_target

        try:
            q_target = self.robot.inverse_kinematics(link=self.ee_link, pos=self.smooth_goal_xyz, quat=np.array([0, 1, 0, 0]))
            q_target[-2:] = gripper_width
            self.robot.control_dofs_position(q_target)
        except:
            pass

# --- シミュレーション実行 ---
def run_vla_simulation():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.5, -1.5, 2.0), camera_lookat=(0.5, 0.0, 0.5)),
        show_viewer=True,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    robot = scene.add_entity(morph=gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    scene.add_entity(morph=gs.morphs.Box(size=(0.06, 0.06, 0.06), pos=(0.5, 0.2, 0.02)), surface=gs.surfaces.Rough(color=(1.0, 0.2, 0.2)))
    
    cameras = [
        scene.add_camera(res=(224, 224), pos=(1.2, 0.15, 0.6), lookat=(0.5, 0.15, 0.0)),
        scene.add_camera(res=(224, 224), pos=(0.5, -0.6, 0.6), lookat=(0.5, 0.15, 0.0)),
        scene.add_camera(res=(224, 224), pos=(0, 0, 1), lookat=(0, 0, 0)),
        scene.add_camera(res=(224, 224), pos=(-0.2, 0.8, 0.8), lookat=(0.5, 0.2, 0.0))
    ]
    scene.build()

    # エージェントと共有メモリの初期化
    bb = Blackboard()
    perception = PerceptionAgent(cameras)
    brain = BrainAgent()
    controller = ControllerAgent(robot, robot.get_link('hand'))

    # 推論エージェントを別スレッドで開始
    threading.Thread(target=perception.run_inference, args=(bb,), daemon=True).start()

    neutral_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])
    
    while True:
        print(f"\n--- TRIAL START ---")
        robot.set_dofs_position(neutral_q)
        robot.set_dofs_kp(np.array([20000] * 9))
        robot.set_dofs_kv(np.array([1000] * 9))
        
        brain.__init__() # 状態リセット
        trial_finished = False

        while not trial_finished:
            scene.step()
            
            # 1. データの収集 (エントメフェクタ位置)
            ee_pos = robot.get_links_pos(controller.ee_link.idx).detach().cpu().numpy().flatten()
            cameras[2].set_pose(pos=ee_pos + np.array([0, 0, 0.03]), lookat=ee_pos + np.array([0, 0, -0.1]))

            # 2. Brainによる意志決定
            state = brain.update_logic(ee_pos, bb.target_pos)

            # 3. Controllerによる実行
            controller.move(state, bb.target_pos)

            # 4. 特殊判定 (LIFT中の成功チェック)
            if state == "LIFT" and (brain.step_count - brain.state_start_step > 800):
                dofs_pos = robot.get_dofs_position().detach().cpu().numpy().flatten()
                if (dofs_pos[-1] + dofs_pos[-2]) / 2.0 > 0.005:
                    print("SUCCESS: Object captured!")
                    return
                else:
                    print("FAILED: Empty grasp. Retrying...")
                    trial_finished = True

if __name__ == "__main__":
    run_vla_simulation()

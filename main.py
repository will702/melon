import kivy

kivy.require('2.0.0')
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
from kivy.utils import platform
from kivy.uix.label import Label
from kivy.graphics import Ellipse

if platform == "android":

    from android.permissions import request_permissions, Permission

    request_permissions([Permission.READ_EXTERNAL_STORAGE,
                         Permission.WRITE_EXTERNAL_STORAGE, Permission.CAMERA])

import filetype
import cv2
import numpy as np
from mymodels import Main_DSS
from mymodels import PreProcessing_frame
import os

im_paths = os.getcwd()

Builder.load_file('design.kv')

class MainScreen(Screen):

    def put_images(self):
        imgs_ids = [self.ids.im_final, self.ids.melon_id,
                    self.ids.melon_mask_id, self.ids.roi_fin_id,
                    self.ids.red_id, self.ids.green_id, self.ids.blue_id,
                    self.ids.index_id]

        for img_id in imgs_ids:
            img_id.reload()

        self.ids.car_ims.remove_widget(self.ids.title_text)
        self.ids.car_ims.add_widget(self.ids.title_text)

class FilechooserScreen(Screen):
    vid_container = None
    vid_show = None
    cvNet = cv2.dnn.readNetFromTensorflow('ssd_melon_model_18853/frozen_inference_graph.pb',
                                          'ssd_melon_model_18853/graph_ori.pbtxt')

    def lsmedia(self, mypath):
        try:
            filename = mypath
            print(filename)
            if filetype.is_image(filename):
                self.ids.img_path.source = filename
                self.ids.img_vid_btn.opacity = '0'
                self.ids.img_vid_btn.disabled = True

            elif filetype.is_video(filename):
                self.ids.img_path.source = r"images/img_none.png"
                with open(r"pathToVid.txt", "w") as f:
                    f.write(filename)
                self.ids.img_vid_btn.opacity = '10'
                self.ids.img_vid_btn.disabled = False

            else:
                self.ids.img_path.source = r"images/img_none.png"
                self.ids.img_vid_btn.opacity = '0'
                self.ids.img_vid_btn.disabled = True
        except:
            pass

    def choose(self):
        from plyer import  filechooser
        '''
        f
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        self.selection = selection[0]
        self.lsmedia(self.selection)

    def selected(self):

        self.choose()
        #print(os.getcwd())
        os.chdir(im_paths)
        #print(os.getcwd())




    def chk(self):

        # self.ids.img_detect_choser.source = 'images/process2.png'
        im = self.ids.img_path
        # im.reload()
        img = cv2.imread(im.source)
        identify_melon = Main_DSS.detect_melon(self.cvNet, img, im_paths)
        if identify_melon == 0:
            pass
        elif identify_melon == 1:
            self.go_to_main()

    def go_to_main(self):
        self.manager.transition.direction = "up"
        self.manager.current = "main_scr"


class VidShow(Screen):
    cvNet = cv2.dnn.readNetFromTensorflow('ssd_melon_model_18853/frozen_inference_graph.pb',
                                          'ssd_melon_model_18853/graph_ori.pbtxt')
    def videoplayer_state(self):
        if self.ids.vid.state == "stop" or self.ids.vid.state == "pause":
            self.ids.on_vid_pause.opacity = 1
            self.ids.on_vid_pause.disabled = False
        else:
            self.ids.on_vid_pause.opacity = 0
            self.ids.on_vid_pause.disabled = True

    def on_vid_touched(self):
        if self.ids.vid.state == "play":
            self.ids.vid.state = "pause"
        else:
            self.ids.vid.state == "stop" or self.ids.vid.state == "pause"
            self.ids.vid.state = "play"

    def read_path(self):
        with open("pathToVid.txt", "r") as fr:
            p = (fr.readlines())[0]
        return p

    def snapShot_from_vid(self, *args):
        im_cap = self.ids.vid.export_as_image()
        w, h = im_cap._texture.size
        frame = np.frombuffer(im_cap._texture.pixels, 'uint8').reshape(h, w, 4)
        opCv_img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        identify_melon = Main_DSS.detect_melon(self.cvNet, opCv_img, im_paths)
        if identify_melon == 0:
            self.ids.vid.state = "play"
        elif identify_melon == 1:
            self.manager.transition.direction = "up"
            self.manager.current = "main_scr"

    def on_vid_pause_press(self):
        im = self.ids.img_process_vid_btn
        im.source = 'images/process2.png'
        im.reload()

    def on_vid_pause_released(self):
        im = self.ids.img_process_vid_btn
        im.source = 'images/process.png'
        im.reload()
        self.snapShot_from_vid()


class RealTimeWin(Screen):
    cameraObject = None
    layout = None
    lbl = None

    cvNet = cv2.dnn.readNetFromTensorflow('ssd_melon_model_18853/frozen_inference_graph.pb',
                                          'ssd_melon_model_18853/graph_ori.pbtxt')

    def cma_play(self):
        self.layout = self.ids.cam_pos
        self.cameraObject = self.ids.camera
        #self.cameraObject.set_landscape(reverse=False)
        self.cameraObject.play = True

        with self.layout.canvas:
            self.bg = Ellipse(source='images/pointer.png',
                              pos=(self.cameraObject.center_x,
                                   self.cameraObject.center_y),
                              size=(50, 50))
        with self.layout.canvas:
            self.lbl = Label(text='', color=(1, 0, 0, 1),
                             pos=(self.cameraObject.center_x,
                                  self.cameraObject.center_y),
                             font_size=20)

        return self.layout

    def get_frame(self, dt):
        self.ids.mask_result.source = 'images/img_none.png'
        self.ids.roi_btn.source = 'images/img_none.png'
        cam = self.cameraObject
        # cam.source = "melon_ATP.mp4"
        image_object = cam.export_as_image()
        w, h = image_object._texture.size
        frame = np.frombuffer(image_object._texture.pixels, 'uint8').reshape(h, w, 4)
        cvImg = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.imwrite("test.png", cvImg)
        img = cv2.resize(cvImg, (int(self.layout.width), int(self.layout.height)))

        rip, ind, blobs = PreProcessing_frame.final_step(self.cvNet, img)

        if blobs > 15:
            b = "(Sick-Melon)"
        else:
            b = ""
        # [[195, 109, 326, 221]]
        if rip == 0:
            self.lbl.text = 'Scanning'
            self.lbl.center_x = self.cameraObject.center_x + 25
            self.lbl.center_y = self.cameraObject.center_y - 25

            self.bg.pos = (self.cameraObject.center_x, self.cameraObject.center_y)

            self.bg.source = 'images/pointer.png'
        else:
            masked_result = "mymodels/temp/masked_result.png"
            roi_fin = "mymodels/temp/roi_fin.png"
            self.ids.mask_result.source = masked_result
            self.ids.roi_btn.source = roi_fin

            if rip == "Ripe":
                self.bg.source = 'images/pointer_ripe.png'
            elif rip == "About to Ripe":
                self.bg.source = 'images/pointer_atr.png'
            elif rip == "Under Ripe":
                self.bg.source = 'images/pointer_ur.png'

            self.lbl.text = '{} {} {}'.format(rip, ind, b)
            self.lbl.center_x = self.cameraObject.center_x + 25
            self.lbl.center_y = self.cameraObject.center_y - 25

            self.bg.pos = (self.cameraObject.center_x, self.cameraObject.center_y)

        Clock.schedule_once(self.get_frame, 0.48)

    def onLeave_vid(self):
        self.ids.cap_btn.state = "normal"
        self.ids.img_cap_btn.source = 'images/pause.png'
        self.ids.back_to_main.source = 'images/back.png'
        self.ids.btn_img_live.source = 'images/live_on.png'
        self.ids.live_btn.state = "normal"

        self.ids.detect_img_btn.opacity = '0'
        self.ids.detect_img_btn.disabled = True
        self.cameraObject.state = "stop"
        self.remove_widget(self.layout)

    def go_to_main(self):
        self.manager.transition.direction = "up"
        self.manager.current = "main_scr"

    def stop_play_state(self):
        if self.ids.cap_btn.state == "down":
            self.cameraObject.play = False
            self.ids.img_cap_btn.source = 'images/play.png'
            # self.ids.cap_btn.text='Resume'
            self.ids.detect_img_btn.opacity = '5'
            self.ids.detect_img_btn.disabled = False

        elif self.ids.cap_btn.state == "normal":
            self.cameraObject.play = True
            self.ids.img_cap_btn.source = 'images/pause.png'
            self.ids.detect_img_btn.opacity = '0'
            self.ids.detect_img_btn.disabled = True

    def snapShot_detection(self, *args):

        im_cap = self.cameraObject.export_as_image()
        w, h = im_cap._texture.size
        frame = np.frombuffer(im_cap._texture.pixels, 'uint8').reshape(h, w, 4)
        opCv_img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        identify_melon = Main_DSS.detect_melon(self.cvNet, opCv_img, im_paths)

        if identify_melon == 0:
            self.cameraObject.play = True
            self.ids.cap_btn.state = "normal"
            self.ids.cap_btn.text = 'Pause'
            self.ids.detect_img_btn.opacity = '0'
            self.ids.detect_img_btn.disabled = True

        elif identify_melon == 1:
            self.manager.transition.direction = "down"
            self.manager.current = "main_scr"

    # self.manager.current = "main_scr"

    def live_off_on(self):
        if self.ids.live_btn.state == "down":
            self.ids.btn_img_live.source = 'images/live_off.png'
            Clock.schedule_once(self.get_frame, 0.05)
            self.ids.detect_img_btn.opacity = "0"
            self.ids.detect_img_btn.disabled = True
            self.ids.cap_btn.opacity = "0"
            self.ids.cap_btn.disabled = True
            self.cameraObject.play = True

        else:
            self.ids.btn_img_live.source = 'images/live_on.png'
            self.ids.mask_result.source = 'images/img_none.png'
            self.ids.roi_btn.source = 'images/img_none.png'
            self.cameraObject.play = True
            self.ids.cap_btn.opacity = "10"
            self.ids.cap_btn.disabled = False
            Clock.unschedule(self.get_frame)
            self.lbl.text = ""
            self.bg.source = ""


class AboutScr(Screen):

    def abs_text(self):
        with open("abstract.txt", "r") as f:
            t = f.read()
        return t


class RootWidget(ScreenManager):
    pass


class MainApp(App):
    def build(self):
        self.icon = 'icon.png'
        return RootWidget()


if __name__ == "__main__":
    MainApp().run()

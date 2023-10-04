from pynput import keyboard


class KeyboardEventHandler:
    def __init__(self,_evt):
        self.listener_thread = keyboard.Listener(on_press=self.isPressed)
        self.listener_thread.start()
        self.is_space_pressed = False
        self.evt = _evt
        self.reset_flag = False
        self.activate = False

        self.training = True
        self.print_param = True
        self.print_action = True

    def isPressed(self,key):
        
        if key == keyboard.Key.esc:
            self.activate = not self.activate
            if self.activate:
                print("Keyboard input Activated")
            else:
                print("Keyboard input Inactivated")
            
        if self.activate:
            if key == keyboard.Key.space:
                if self.is_space_pressed:
                    self.is_space_pressed = False
                    print("GO")
                else:
                    self.is_space_pressed = True
                    print("STOP")
            
            if key == keyboard.KeyCode(char='t'):
                self.training = not self.training
           
            if key == keyboard.KeyCode(char='r'):
                self.reset_flag = True

            if key == keyboard.KeyCode(char='p'):
                self.print_param = not self.print_param
            
            if key == keyboard.KeyCode(char='a'):
                self.print_action = not self.print_action
    
    def isTrainingMode(self):
        return self.training
    
    def isPrintParam(self):
        return self.print_param
    
    def isPrintAction(self):
        return self.print_action
                
                

            



     
# evt = threading.Event()
# a = KeyboardEventHandler()


# while True:
#     #print(0)
#     evt.wait()
#     #print(1)
#     evt.clear()


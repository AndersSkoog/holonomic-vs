import pyglet
from utils import proj_file_path

def pyglet_image(path:str,filename:str):
    pyglet.resource.path = proj_file_path(path).split("/")
    return pyglet.resource.image(filename)

pyglet.resource.path = proj_file_path("/img").split("/")
bar = pyglet.resource.image("bar.png")
knob = pyglet.resource.image("knob.png")
tog_on = pyglet.resource.image("pressed.png")
tog_off = pyglet.resource.image("unpresed.png")
tog_hov = pyglet.resource.image("hover.png")

class WidgetFrame:
  def __init__(
    self,
    win:pyglet.window.Window,
    prog:pyglet.graphics.shader.ShaderProgram,
    prog_batch:pyglet.graphics.Batch,
    uniform_update:callable,
    frame_pos:(int,int),
    params
  ):
    self.window = win
    self.batch = prog_batch
    self.frame_pos = frame_pos
    self._prog = prog
    self._uniform_update = uniform_update
    self._cell_margin = 20
    self._cell_size = 64
    self._frame = pyglet.gui.Frame(window=self.window,cell_size=self._cell_size,order=4)
    #self._slider_bar = Image.new("RGB",(self._cell_size*4,self._cell_size),(255,0,0)) #red slider bar
    #self._slider_knob = Image.new("RGB",(25,25),(0,0,255)) # blue slider knob
    #self._toggle_pressd = Image.new("RGB",(25,25),(0,255,0)) #turn green when pressed
    #self._toggle_unpressed = Image.new(mode="RGB",size=(25,25),color=(100,100,100)) #turn grey when unpressed
    #self._toggle_hover = Image.new(mode="RGB",size=(30,30),color=(200,200,200)) #increase size slighly and make white when hovered
    
    self.params = params  # key -> current value of each parameter
    self.labels = {}  # key -> pyglet.text.Label showing parameter
    self.widgets = {}  # key -> widget instance

  def _calc_wid_pos(self,wid_index):
      x,y,m,cs = self.frame_pos[0],self.frame_pos[1],self._cell_margin,self._cell_size
      ret_x = x
      ret_y = y - (m + ((cs+m)*wid_index))
      return ret_x,ret_y

  def _calc_label_pos(self,wid_index):
      wid_pos = self._calc_wid_pos(wid_index)
      return wid_pos[0],wid_pos[1] + self._cell_margin // 2


  def _slider_handler(self,param_name:str,parser:callable):
    def handler(widget_value):
      parsed_value = parser(widget_value)
      self.params[param_name] = parsed_value
      self.labels[param_name].text = f"{param_name}: {parsed_value:.3f}"
      self._uniform_update(self.params,self._prog)
      #self._prog[uniform_key] = uniform_gen(**self.params)
    return handler

  def _toggle_handler(self,param_name:str,parser:callable):
    def handler(widget_state):
      parsed_value = parser(widget_state)
      self.params[param_name] = parsed_value
      self.labels[param_name].text = f"{param_name}: {parsed_value}"
      self._uniform_update(self.params,self._prog)
    return handler

  def _input_handler(self,param_name:str,parser:callable):
    def handler(widget_text):
      parsed_value = parser(widget_text)
      self.params[param_name] = parsed_value
      self._uniform_update(self.params,self._prog)
    return handler

    # -----------------------------
    # Registration methods
    # -----------------------------
  def reg_slider(self,param_name:str,init_val,parser:callable):
    wid_cnt = len(list(self.widgets.keys()))
    wid_ind = wid_cnt
    wid_pos = self._calc_wid_pos(wid_ind)
    label_pos = self._calc_label_pos(wid_ind)
    label = pyglet.text.Label(
        text=f"{param_name}:{init_val}",
        x=label_pos[0],y=label_pos[1],
        width=self._cell_size,height=self._cell_margin//2,
        color=(0,0,0,255)
    )
    wid = pyglet.gui.Slider(
        x=wid_pos[0],y=wid_pos[1],
        base=bar,knob=knob,
        edge=5,batch=self.batch
    )
    wid.set_handler('on_change',self._slider_handler(param_name,parser))
    self.params[param_name] = init_val
    self.labels[param_name] = label
    self.widgets[param_name] = wid
    self._frame.add_widget(self.widgets[param_name])
    return True

  def reg_toggle(self, param_name:str,parser:callable):
    wid_cnt = len(list(self.widgets.keys()))
    wid_ind = wid_cnt
    wid_pos = self._calc_wid_pos(wid_ind)
    label_pos = self._calc_label_pos(wid_ind)
    label = pyglet.text.Label(
      text=f"{param_name}:{0}",
      x=label_pos[0], y=label_pos[1],
      width=self._cell_size, height=self._cell_margin // 2,
      color=(0, 0, 0, 255)
    )
    wid = pyglet.gui.ToggleButton(
        x=wid_pos[0], y=wid_pos[1],
        pressed=tog_on,unpressed=tog_off,hover=tog_hov,
        batch=self.batch
    )
    wid.set_handler('on_toggle',self._toggle_handler(param_name,parser))
    self.params[param_name] = 0
    self.labels[param_name] = label
    self.widgets[param_name] = wid
    self._frame.add_widget(self.widgets[param_name])
    return True

  def reg_input(self,param_name:str,init_val,parser:callable):
    wid_cnt = len(list(self.widgets.keys()))
    wid_ind = wid_cnt
    wid_pos = self._calc_wid_pos(wid_ind)
    label_pos = self._calc_label_pos(wid_ind)
    label = pyglet.text.Label(
      text=f"{param_name}",
      x=label_pos[0], y=label_pos[1],
      width=self._cell_size, height=self._cell_margin // 2,
      color=(0, 0, 0, 255)
    )
    wid = pyglet.gui.TextEntry(
      text="",
      x=wid_pos[0], y=wid_pos[1],batch=self.batch,width=64*4,
      color=(0,0,0,255),text_color=(255,255,255,255),caret_color=(255,255,255,255)
    )
    wid.set_handler('on_commit', self._input_handler(param_name,parser))
    self.params[param_name] = init_val
    self.labels[param_name] = label
    self.widgets[param_name] = wid
    self._frame.add_widget(self.widgets[param_name])
    return True


""""
img_path = proj_file_path("/img")
slider_bar = Image.new(mode="RGB",size=(64*4,64),color=(255, 0, 0))  # red slider bar
slider_knob = Image.new(mode="RGB",size=(64, 64),color=(0, 0, 255))  # blue slider knob
toggle_pressed = Image.new(mode="RGB",size=(64, 64),color=(0, 255, 0))  # turn green when pressed
toggle_unpressed = Image.new(mode="RGB",size=(64, 64),color=(100, 100, 100))  # turn grey when unpressed
toggle_hover = Image.new(mode="RGB", size=(64, 64),color=(200, 200, 200))  # turn white when hovered
slider_bar.save(img_path+"/bar.png","png")
slider_knob.save(img_path+"/knob.png","png")
toggle_pressed.save(img_path+"/pressed.png","png")
toggle_unpressed.save(img_path+"/unpresed.png","png")
toggle_hover.save(img_path+"/hover.png","png")
"""
#img_path = proj_file_path("/img").split("/")
#pyglet.resource.path = img_path

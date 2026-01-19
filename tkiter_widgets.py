from math import tau
import tkinter as tk
from tkinter import ttk
from tkinter.constants import SINGLE
from typing import Sequence
from PlotContext import PlotContext
#from datafile import DataManager

def parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]


class SelectBox(ttk.Frame):
    def __init__(self, ctx:PlotContext, wid_id:str, label_text:str,options,on_change,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.plotctx = ctx
        self.wid_id = wid_id
        self.var = tk.StringVar(value=options[0])
        ttk.Label(ctx.widget_frame, text=label_text).pack(anchor="w")
        ttk.OptionMenu(ctx.widget_frame, self.var, self.var.get(), *options).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write", self._changed)
        self.on_change = on_change
        self.pack(fill=tk.X, pady=1)


    def _changed(self, *args):
        cur_value = self.var.get()
        self.on_change(self.wid_id,cur_value)

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)

    def set_silent(self,v:str):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(v)
        self._trace_id = self.var.trace_add("write", self._changed)

class ListBox(ttk.Frame):
    def __init__(
        self,
        ctx,
        wid_id: str,
        label_text: str,
        list_data,
        on_change,
        **kwargs
    ):
        super().__init__(ctx.widget_frame, **kwargs)

        self.plotctx = ctx
        self.wid_id = wid_id
        self.on_change = on_change

        # Label
        ttk.Label(self, text=label_text).pack(anchor="w")

        # Variable holding list items
        self.var = tk.Variable(value=list(list_data))

        # Listbox
        self.listbox = tk.Listbox(
            self,
            listvariable=self.var,
            selectmode=tk.SINGLE,
            exportselection=False
        )
        self.listbox.pack(fill=tk.X)

        # Bind selection event
        self.listbox.bind("<<ListboxSelect>>", self._changed)

        self.pack(fill=tk.X, pady=1)

    def _changed(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        value = self.listbox.get(sel[0])
        self.on_change(self.wid_id, value)

    def get(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        return self.listbox.get(sel[0])

    def set(self, value):
        items = self.var.get()
        if value in items:
            idx = items.index(value)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
            self.on_change(self.wid_id, value)

    def set_silent(self, value):
        items = self.var.get()
        if value in items:
            idx = items.index(value)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)


class IntSlider(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str, label_text:str, min_val:int,max_val:int,init_val:int,on_change, **kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.wid_id  = wid_id
        self.var = tk.IntVar(value=init_val)
        ttk.Label(ctx.widget_frame, text=label_text).pack(side=tk.TOP, anchor="w")
        tk.Scale(
            master=ctx.widget_frame,from_=min_val,to=max_val,
            orient="horizontal",variable=self.var,showvalue=True
        ).pack(fill=tk.X)
        self.on_change = on_change
        self._trace_id = self.var.trace_add("write",self._changed)
        self.pack(fill=tk.X, pady=1)


    def _changed(self,*args):
        cur_value = self.var.get()
        if self.on_change:
            self.on_change(self.wid_id,cur_value)

    def get(self):
        #Return current selection.
        return self.var.get()

    def set(self, value:int):
        #Set current selection.
        self.var.set(value)

    def set_silent(self,value:int):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(value)
        self._trace_id = self.var.trace_add("write", self._changed)


class FloatSlider(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str, label_text:str, min_val:float,max_val:float,init_val:float,on_change,res=0.01, **kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.wid_id  = wid_id
        self.var = tk.DoubleVar(value=init_val)
        self.on_change = on_change
        ttk.Label(master=ctx.widget_frame,text=label_text).pack(side=tk.TOP, anchor="w")
        tk.Scale(master=ctx.widget_frame,from_=min_val,to=max_val,orient="horizontal",variable=self.var,showvalue=True,resolution=res).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write",self._changed)
        self.pack(fill=tk.X, pady=1)

    def _changed(self, *args):
        cur_value = float(self.var.get())
        if self.on_change:
            self.on_change(self.wid_id,cur_value)

    def get(self):
        #Return current selection.
        return self.var.get()

    def set(self, value:float):
        #Set current selection.
        self.var.set(value)

    def set_silent(self,value:float):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(value)
        self._trace_id = self.var.trace_add("write", self._changed)


class Angle(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str, label_text:str,step_cnt:int,on_change,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.step_val = tau / step_cnt
        self.wid_id  = wid_id
        self.var = tk.StringVar(value="0.0")
        self.on_change = on_change
        ttk.Label(ctx.widget_frame, text=label_text).pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=0.0,to=tau,increment=self.step_val,format='%8.4f',textvariable=self.var).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write", self._changed)
        self.pack(fill=tk.X, pady=1)


    def _changed(self, *args):
        cur_value = float(self.var.get())
        if self.on_change:
            self.on_change(self.wid_id, cur_value)

    def get(self):
        #Return current selection.
        return float(self.var.get())

    def set(self, value:float):
        #Set current selection.
        self.var.set(str(value))

    def set_silent(self,value:float):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(str(value))
        self._trace_id = self.var.trace_add("write", self._changed)



class StepedRange(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str, label_text:str,center_val,step_val,cnt:int,on_change,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        assert type(center_val) == float or int, "center val must be float or int"
        self.min_val = int(center_val - (step_val * cnt)) if type(center_val) == int else float(center_val - (step_val * cnt))
        self.max_val = int(center_val + (step_val * cnt)) if type(center_val) == int else float(center_val + (step_val * cnt))
        self.wid_id  = wid_id
        self.var = tk.StringVar(value=f"{center_val}")
        self.get_val = lambda: float(self.var.get()) if type(center_val) == "float" else lambda: int(self.var.get())
        self.on_change = on_change
        ttk.Label(ctx.widget_frame, text=label_text).pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=self.min_val,to=self.max_val,increment=step_val,format='%8.4f',textvariable=self.var).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write", self._changed)
        self.pack(fill=tk.X, pady=1)

    def _changed(self, *args):
        cur_value = self.get_val()
        if self.on_change:
            self.on_change(self.wid_id, cur_value)

    def get(self):
        #Return current selection.
        return self.get_val()

    def set(self, value):
        #Set current selection.
        self.var.set(str(value))

    def set_silent(self,value):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(str(value))
        self._trace_id = self.var.trace_add("write", self._changed)

class NumberboxInt(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str, label_text:str,on_change,init_val:int,max_val=10000,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.wid_id  = wid_id
        self.var = tk.StringVar(value="0")
        self.on_change = on_change
        ttk.Label(ctx.widget_frame, text=label_text).pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=0,to=max_val,increment=1,textvariable=self.var).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write", self._changed)
        self.pack(fill=tk.X, pady=1)

    def _changed(self, *args):
        cur_value = int(self.var.get())
        if self.on_change:
            self.on_change(self.wid_id, cur_value)

    def get(self):
        #Return current selection.
        return int(self.var.get())

    def set(self, value):
        #Set current selection.
        self.var.set(str(value))

    def set_silent(self,value):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(str(value))
        self._trace_id = self.var.trace_add("write", self._changed)

class NumberListEntry(ttk.Frame):
    def __init__(self, ctx:PlotContext,wid_id:str,label_text:str,init_val,num_type,on_change,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        assert num_type in ("float","int"), "not valid numtype"
        self.wid_id = wid_id
        self.num_type = num_type
        self.var = tk.StringVar(value=','.join(map(str,init_val)))
        self.on_change = on_change
        ttk.Label(ctx.widget_frame, text=label_text).pack(side=tk.TOP, anchor="w")
        ttk.Entry(master=ctx.widget_frame,textvariable=self.var).pack(fill=tk.X)
        self._trace_id = self.var.trace_add("write", self._changed)
        self.pack(fill="x", pady=2)

    def _changed(self, *args):
        text = self.var.get()
        try:
            parts = text.replace(","," ").split()
            nums = [int(p) for p in parts] if self.num_type == "int" else [float(p) for p in parts]
            self.on_change(self.wid_id, nums)
        except ValueError: raise ValueError("gsdgas")

    def get(self):
        s = self.var.get().split(",")
        return list(map(float,s))

    def set(self, value:Sequence[float]):
        self.var.set(','.join(map(str,value)))

    def set_silent(self, value):
        self.var.trace_remove("write", self._trace_id)
        self.var.set(','.join(map(str,value)))
        self._trace_id = self.var.trace_add("write", self._changed)
"""
class PresetCtrl(ttk.Frame):
    def __init__(self, ctx:PlotContext,presetfilename:str,get_params_callback,apply_params_callback,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.preset_manager = PresetManager(presetfilename)
        self.presets_names = self.preset_manager.list_presets()
        self.selboxvar = tk.StringVar(value="none")
        self.get_params = get_params_callback
        self.apply_params = apply_params_callback
        ttk.Label(ctx.widget_frame, text="select preset").pack(anchor="w")
        self.selbox = ttk.Combobox(
            ctx.widget_frame,
            textvariable=self.selboxvar,
            values=self.presets_names,
            state="readonly"
        )
        self.selbox.pack(fill=tk.X)
        ttk.Button(ctx.widget_frame, text="Save Preset", command=self._save_preset).pack(fill=tk.X)
        ttk.Button(ctx.widget_frame, text="Load Preset", command=self._load_preset).pack(fill=tk.X)
        self.pack(fill="x", pady=2)

    def _refresh_presets_list(self):
        self.presets_names = self.preset_manager.list_presets()
        self.selbox["values"] = self.presets_names
        self.selboxvar.set(self.presets_names[-1])  # select newest


    def _save_preset(self):
        name = simpledialog.askstring("Save Preset", "Preset name?")
        if not name:
            return
        params = self.get_params()
        self.preset_manager.save_preset(name, params)
        self._refresh_presets_list()
        messagebox.showinfo("Saved", f"Preset '{name}' saved.")

    def _load_preset(self):
        preset = self.selboxvar.get()
        if preset in self.presets_names:
            params = self.preset_manager.load_preset(preset)
            self.apply_params(params)
            
class FourierCurveCtrl(ttk.Frame):
    def __init__(self, ctx:PlotContext,presetfilename:str,apply_cb,**kwargs):
        super().__init__(ctx.widget_frame, **kwargs)
        self.apply_cb = apply_cb
        self.preset_manager = PresetManager(presetfilename)
        self.presets_names = self.preset_manager.list_presets()
        self.cur_preset = self.presets_names[0]
        self.cur_params = self.preset_manager.load_preset(self.cur_preset)
        self.var_preset = tk.StringVar(value=self.cur_preset)
        self.var_coef_cnt = tk.StringVar(value=str(self.cur_params["coef_cnt"]))
        self.var_m = tk.StringVar(value=str(self.cur_params["M"]))
        self.var_l = tk.StringVar(value=str(self.cur_params["L"]))
        self.var_coef_a   = tk.StringVar(value=','.join(map(str,self.cur_params["coef_a"])))
        self.var_coef_b   = tk.StringVar(value=','.join(map(str,self.cur_params["coef_b"])))
        self.var_coef_phi = tk.StringVar(value=','.join(map(str,self.cur_params["coef_phi"])))
        self.var_coef_psi = tk.StringVar(value=','.join(map(str,self.cur_params["coef_psi"])))
        self.var_res = tk.StringVar(value=str(self.cur_params["res"]))

        ttk.Label(ctx.widget_frame, text="coef_count").pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=1,to=12,increment=1,textvariable=self.var_coef_cnt).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="res").pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=1,to=12,increment=1,textvariable=self.var_res).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="M").pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=1,to=12,increment=1,textvariable=self.var_m).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="L").pack(side=tk.TOP, anchor="w")
        ttk.Spinbox(master=ctx.widget_frame,from_=1,to=12,increment=1,textvariable=self.var_l).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="coef_a").pack(side=tk.TOP, anchor="w")
        ttk.Entry(master=ctx.widget_frame,textvariable=self.var_coef_a).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="coef_b").pack(side=tk.TOP, anchor="w")
        ttk.Entry(master=ctx.widget_frame,textvariable=self.var_coef_b).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="coef_phi").pack(side=tk.TOP, anchor="w")
        ttk.Entry(master=ctx.widget_frame,textvariable=self.var_coef_phi).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="coef_psi").pack(side=tk.TOP, anchor="w")
        ttk.Entry(master=ctx.widget_frame,textvariable=self.var_coef_psi).pack(fill=tk.X)
        ttk.Label(ctx.widget_frame, text="select preset").pack(anchor="w")
        self.selbox = ttk.Combobox(ctx.widget_frame,textvariable=self.var_preset,values=self.presets_names,state="readonly")
        self.selbox.pack(fill=tk.X)
        ttk.Button(ctx.widget_frame, text="Save Preset", command=self._save_preset).pack(fill=tk.X)
        ttk.Button(ctx.widget_frame, text="Load Preset", command=self._load_preset).pack(fill=tk.X)
        self.var_coef_cnt.trace_add("write",self._update_coef_cnt)
        self.var_l.trace_add("write",self._update_params)
        self.var_m.trace_add("write",self._update_params)
        self.var_coef_a.trace_add("write",self._update_params)
        self.var_coef_b.trace_add("write",self._update_params)
        self.var_coef_phi.trace_add("write",self._update_params)
        self.var_coef_psi.trace_add("write",self._update_params)

    def _update_params(self, *args):
        try:
            coef_a = parse_floats(self.var_coef_a.get())
            coef_b = parse_floats(self.var_coef_b.get())
            coef_phi = parse_floats(self.var_coef_phi.get())
            coef_psi = parse_floats(self.var_coef_psi.get())

            # Prevent incomplete updates
            lens = {len(coef_a), len(coef_b), len(coef_phi), len(coef_psi)}
            if len(lens) != 1:
                return

            self.cur_params["res"] = int(self.var_res.get())
            self.cur_params["L"] = int(self.var_l.get())
            self.cur_params["M"] = int(self.var_m.get())
            self.cur_params["coef_cnt"] = int(self.var_coef_cnt.get())
            self.cur_params["coef_a"] = coef_a
            self.cur_params["coef_b"] = coef_b
            self.cur_params["coef_phi"] = coef_phi
            self.cur_params["coef_psi"] = coef_psi

            self.apply_cb(self.cur_params)

        except ValueError:
            print("invalid input")
            return

    def _update_coef_cnt(self,*args):
        v = int(self.var_coef_cnt.get())
        for k in ["coef_a","coef_b","coef_phi","coef_psi"]:
            li = self.cur_params[k]
            l = len(li)
            if v > l:
                zeros = [0.0] * (v-l)
                self.cur_params[k] = li + zeros
            if v < l:
                self.cur_params[k] = li[0:(l-v)]
        self.apply_cb(self.cur_params)



    def _refresh_presets_list(self):
        self.presets_names = self.preset_manager.list_presets()
        self.selbox["values"] = self.presets_names
        self.var_preset.set(self.presets_names[-1])  # select newest


    def _save_preset(self):
        name = simpledialog.askstring("Save Preset", "Preset name?")
        if not name:
            return
        self.preset_manager.save_preset(name, self.cur_params)
        self._refresh_presets_list()
        messagebox.showinfo("Saved", f"Preset '{name}' saved.")

    def _load_preset(self):
        preset = self.var_preset.get()
        self.cur_params = self.preset_manager.load_preset(preset)
        self.apply_cb(self.cur_params)
"""

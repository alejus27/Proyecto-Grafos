import itertools
import tkinter
from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter.filedialog import *

from PIL import ImageTk

from Grafos import *
from Canvas import *
import random
import pickle
import re
import json
import ast
from pprint import pprint
import pandas as pd
import pandas
import os
import tempfile
import win32api
import win32print
from pymongo import MongoClient
import pyscreenshot as ImageGrab
from ctypes import windll
import ast
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import pymongo
from bson.json_util import dumps, loads
from datetime import datetime
from PIL import Image, ImageTk
import webbrowser
import numpy as np
from functools import partial

user32 = windll.user32
user32.SetProcessDPIAware()


class Window:
    def __init__(self, g=None):
        self.root = Tk()
        self.root.title("*** GRAFOS ***")
        # self.root.eval('tk::PlaceWindow . center')

        self.root.option_add("*Font", 'Calibri 10')

        # self.root.configure(bg='black')

        self.g = g

        self.default_col_v_fill = StringVar()
        self.default_col_v_outline = StringVar()
        self.default_col_r = StringVar()
        self.default_col_weight = StringVar()
        self.default_v_size = IntVar()
        self.width = IntVar()
        self.height = IntVar()

        self.directed = IntVar()
        self.multigraph = IntVar()
        self.weighted = IntVar()

        self.directed.set(1)
        self.multigraph.set(1)
        self.weighted.set(1)
        # self.creating = 0
        self.names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.next_name = None

        self.next = None
        self.cancel_stack = []
        self.event = None

        self.ask_vers = IntVar()
        self.ask_ribs = IntVar()
        self.entry = None

        self.history = []

        self.aux = False;

        self.n=[]
        self.a=[]

        self.img_n = PhotoImage(file='icons/kub-approve.png')

        self._make_widgets()
        mainloop()

    def _make_widgets(self):
        self._make_menu()

        self._make_top_frame()
        self._make_create_frame()

        # self.canvas_frame = Frame(self.root)
        # self.canvas_frame.grid(row = 2, column = 0)
        self.canvas = GraphCanvas(self.root, width=1000, height=600, scrollregion=(0, 0, 600, 400), bg="#D3D3D3",
                                  highlightbackground="black")  ##
        self.canvas.grid(row=1, column=0, sticky=N + S + E + W)
        self.y_scrollbar = Scrollbar(self.root, command=self.canvas.yview, orient=VERTICAL)
        self.x_scrollbar = Scrollbar(self.root, command=self.canvas.xview, orient=HORIZONTAL)
        self.canvas.configure(yscrollcommand=self.y_scrollbar.set, xscrollcommand=self.x_scrollbar.set)

        self.y_scrollbar.grid(row=1, column=1, columnspan=1, sticky='ns')
        self.x_scrollbar.grid(row=2, column=0, columnspan=1, sticky='ew')

        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        if self.g:
            self.canvas.view_graph(self.g)

        self._make_vertex_popup()
        self._make_rib_popup()

    def _make_menu(self):


        self.img_nuevo = PhotoImage(file='icons/kub-edit.png')
        self.img_guardar=PhotoImage(file='icons/kub-download.png')
        self.img_abrir = PhotoImage(file='icons/kub-upload.png')
        self.img_exportar = PhotoImage(file='icons/kub-document.png')
        self.img_imprimir = PhotoImage(file='icons/kub-link.png')
        self.img_inicio = PhotoImage(file='icons/kub-home.png')
        self.img_cerrar = PhotoImage(file='icons/kub-remove.png')

        self.img_editar = PhotoImage(file='icons/kub-edit.png')
        self.img_borrar = PhotoImage(file='icons/kub-trash.png')

        self.img_cerrar = PhotoImage(file='icons/kub-remove.png')
        self.img_cerrar = PhotoImage(file='icons/kub-remove.png')
        self.img_cerrar = PhotoImage(file='icons/kub-remove.png')

        self.img_ayuda = PhotoImage(file='icons/kub-help.png')

        self.menubar = Menu(self.root)

        self.file_menu = Menu(self.menubar, tearoff=0)

        # self.file_menu.add_command(label='NUEVO GRAFO', command=self.create_graph)

        self.sub_menu2 = Menu(self.file_menu, tearoff=0)
        self.sub_menu2.add_command(label='PERSONALIZADO',
                                   command=lambda: [self._make_properties_frame(), self.m_create_graph()
                                                    ])

        self.sub_menu2.add_command(label='ALEATORIO', command=self._make_properties_frame2)

        self.file_menu.add_cascade(
            label="NUEVO",
            menu=self.sub_menu2,
            image=self.img_nuevo, compound='left'
        )

        self.sub_menu3 = Menu(self.file_menu, tearoff=0)
        self.sub_menu3.add_command(label='BASE DE DATOS', command=self.mongo)
        self.sub_menu3.add_command(label='JSON', command=self.importar)

        self.file_menu.add_cascade(
            label="ABRIR",
            menu=self.sub_menu3,
            image=self.img_abrir, compound='left'
        )

        # self.file_menu.add_command(label='IMPORTAR (JSON)', command=self.importar)

        # self.file_menu.add_command(label='ABRIR GRAFO', command=self.abrir)

        self.sub_menu4 = Menu(self.file_menu, tearoff=0)
        self.sub_menu4.add_command(label='BASE DE DATOS', command=self.savegrafo)
        self.sub_menu4.add_command(label='JSON', command=self.guardarjson)

        self.file_menu.add_cascade(
            label="GUARDAR",
            menu=self.sub_menu4,
            image=self.img_guardar, compound='left'
        )

        # self.file_menu.add_command(label='GUARDAR GRAFO', command=self.savegrafo)
        # self.file_menu.add_command(label='GUARDAR (JSON)', command=self.guardarjson)

        self.sub_menu = Menu(self.file_menu, tearoff=0)
        self.sub_menu.add_command(label='EXCEL', command=self.saveexcel)
        self.sub_menu.add_command(label='IMAGEN', command=self.saveimage)
        self.sub_menu.add_command(label='PDF', command=self.savepdf)

        self.file_menu.add_cascade(
            label="EXPORTAR DATOS",
            menu=self.sub_menu,
            image=self.img_exportar, compound='left'
        )

        self.file_menu.add_command(label='IMPRIMIR', command=self.imprimir,
            image=self.img_imprimir, compound='left')
        self.file_menu.add_command(label='INICIO', command=self.inicio,
            image=self.img_inicio, compound='left')
        self.file_menu.add_command(label='CERRAR', command=self.root.quit,
                                   image=self.img_cerrar, compound='left'
                                   )

        self.menubar.add_cascade(label='ARCHIVO', menu=self.file_menu)

        self.edit_menu = Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label='EDITAR GRAFO', command=self.edit_graph,
                                   image=self.img_editar, compound='left')
        self.edit_menu.add_command(label='BORRAR GRAFO', command=self.clear_all,
                                   image=self.img_borrar, compound='left')
        self.menubar.add_cascade(label='EDITAR', menu=self.edit_menu)

        self.edit_menu = Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label='ALGORITMO 1')
        self.edit_menu.add_command(label='ALGORITMO 2')
        self.edit_menu.add_command(label='ALGORITMO 3')
        self.menubar.add_cascade(label='ANALIZAR (No implementado aún)', menu=self.edit_menu)

        self.edit_menu = Menu(self.menubar, tearoff=0)
        #self.edit_menu.add_command(label='GRAFICA')
        self.edit_menu.add_command(label='TABLA', command=self.pp)
        self.menubar.add_cascade(label='VENTANA', menu=self.edit_menu)

        self.edit_menu = Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label='MANUAL DE USUARIO',
                                   image=self.img_ayuda, compound='left', command=self.website)
        self.edit_menu.add_command(label='ACERCA DE GRAFOS',
                                   image=self.img_ayuda, compound='left', command=self.website2)
        self.menubar.add_cascade(label='AYUDA', menu=self.edit_menu)

        self.root.config(menu=self.menubar)

    def _make_top_frame(self):
        self.top_frame = Frame(self.root)
        self.top_frame.grid(row=0, column=0, sticky=(W))

        # self._make_buttons_frame()
        # self._make_properties_frame()
        # self._make_configure_frame()
    #1
    def _make_buttons_frame(self):


        self.buttons_frame = Frame(self.top_frame)
        self.buttons_frame.grid(row=0, column=0)
        self.create_button = Button(self.buttons_frame, text='CREAR GRAFO', command=self.create_graph)
        self.create_button.grid(row=0, column=0)
        self.random_button = Button(self.buttons_frame, text='CREAR GRAFO ALEATORIO', command=self.random_graph)
        self.random_button.grid(row=0, column=1)

        self.save_button = Button(self.buttons_frame, text='GUARDAR', command=self.save)
        self.save_button.grid(row=0, column=2)
        self.download_button = Button(self.buttons_frame, text='ABRIR', command=self.download)
        self.download_button.grid(row=0, column=3)

        self.download_button = Button(self.buttons_frame, text='ALGORITMOS', command=self.download)
        self.download_button.grid(row=0, column=4)

    #9

    def _make_properties_frame2(self):

        self.create_frame.grid_remove()

        if (self.aux):
            self.configure_frame.grid_remove()
        properties_frame = Frame(self.top_frame, height=20)

        properties_frame.grid(row=1, column=5)

        l = Label(properties_frame, text="Seleccione el tipo de grafo: ")
        directed_check = Checkbutton(properties_frame, variable=self.directed, text='DIRIGIDO')
        weighted_check = Checkbutton(properties_frame, variable=self.weighted, text='PONDERADO')
        multigraph_check = Checkbutton(properties_frame, variable=self.multigraph, text='MULTIGRAFO')


        b = Button(properties_frame, text="OK", bg='black', fg='white', command=lambda: [self.random_graph(),
                                                                                             properties_frame.grid_remove()])

        l.grid(row=2, column=0)
        directed_check.grid(row=2, column=1)
        weighted_check.grid(row=2, column=2)
        multigraph_check.grid(row=2, column=3)
        b.grid(row=2, column=4)

    def _make_properties_frame(self):

        self.create_frame.grid_remove()

        if(self.aux):
            self.configure_frame.grid_remove()

        properties_frame = Frame(self.top_frame, height=20)

        properties_frame.grid(row=1, column=5)

        l = Label(properties_frame, text="Seleccione el tipo de grafo: ")
        directed_check = Checkbutton(properties_frame, variable=self.directed, text='DIRIGIDO')
        weighted_check = Checkbutton(properties_frame, variable=self.weighted, text='PONDERADO')
        multigraph_check = Checkbutton(properties_frame, variable=self.multigraph, text='MULTIGRAFO')

        #self.img_n = PhotoImage(file='icons/kub-approve.png')

        b = Button(properties_frame,  text="OK",  bg='black', fg='white', font='sans 8 bold', command=lambda: [self.create_graph(),
                                                                                         properties_frame.grid_remove()])

        l.grid(row=2, column=0)
        directed_check.grid(row=2, column=1)
        weighted_check.grid(row=2, column=2)
        multigraph_check.grid(row=2, column=3)
        b.grid(row=2, column=4)



    def _make_configure_frame(self):

        self.aux=True

        self.configure_frame = Frame(self.top_frame)
        self.configure_frame.grid(row=1, column=8, sticky='wn')

        # c1 = random_color() #cambio
        # c2 = random_color()
        c1 = '#000000'
        c2 = '#FF0040'
        self.default_col_v_fill.set(c1)
        self.default_col_v_outline.set(c2)

        self.default_col_weight.set(c1)
        self.default_col_r.set(c2)

        self.default_v_size.set(20)
        # self.width.set(600)
        # self.height.set(400)

        self.vertex_icon = Canvas(self.configure_frame, width=20, height=20)
        self.vertex_icon.create_oval(2, 2, 20, 20)
        self.vertex_icon.create_text(11, 11, text='A')
        self.vertex_icon.grid(row=0, column=1)
        self.vertex_size_entry = Entry(self.configure_frame, textvariable=self.default_v_size, width=3)
        self.vertex_size_entry.grid(row=0, column=2)
        self.vertex_color_button = Canvas(self.configure_frame, width=20, height=20, bg=self.default_col_v_fill.get())
        self.outline_color_button = Canvas(self.configure_frame, width=20, height=20,
                                           bg=self.default_col_v_outline.get())
        self.vertex_color_button.grid(row=0, column=4)
        self.outline_color_button.grid(row=0, column=3)

        self.rib_icon = Canvas(self.configure_frame, width=20, height=20)
        self.rib_icon.create_line(0, 20, 20, 0, arrow=LAST)
        self.rib_icon.grid(row=1, column=1)
        self.rib_color_button = Canvas(self.configure_frame, width=20, height=20, bg=self.default_col_r.get())
        self.weight_color_button = Canvas(self.configure_frame, width=20, height=20, bg=self.default_col_weight.get())
        self.rib_color_button.grid(row=1, column=2)
        self.weight_color_button.grid(row=1, column=3)

        self.l=Label(self.configure_frame, text="    Propiedades del nodo:   ");
        self.l.grid(row=0, column=0)
        self.l = Label(self.configure_frame, text="   Propiedades del arco:   ");
        self.l.grid(row=1, column=0)

        # self.width_entry = Entry(self.configure_frame, textvariable = self.width, width = 5)
        # self.height_entry = Entry(self.configure_frame, textvariable=self.height, width = 5)
        # self.width_entry.grid(row = 0, column = 7)
        # self.height_entry.grid(row = 0, column = 8)

        self.vertex_color_button.bind('<Button-1>',
                                      lambda ev: self.set_color(self.default_col_v_fill, self.vertex_color_button))
        self.outline_color_button.bind('<Button-1>',
                                       lambda ev: self.set_color(self.default_col_v_outline, self.outline_color_button))
        self.rib_color_button.bind('<Button-1>', lambda ev: self.set_color(self.default_col_r, self.rib_color_button))
        self.weight_color_button.bind('<Button-1>',
                                      lambda ev: self.set_color(self.default_col_weight, self.weight_color_button))
        # self.width_entry.bind('<Return>', lambda ev: self.canvas.configure(width = self.width.get(), scrollregion = (0, 0, self.width.get(), self.height.get())))
        # self.height_entry.bind('<Return>', lambda ev: self.canvas.configure(height=self.height.get(), scrollregion = (0, 0, self.width.get(), self.height.get())))

    #99
    def _make_create_frame(self):



        self.create_frame = Frame(self.top_frame)


        # self.create_frame.grid(row=1, column=1, sticky='wn')

        # self.label1 = Label(self.create_frame, text='ALEATORIO', font='Calibri 10 bold underline')
        # self.label1.grid(row=3, column=0)

        # self.add_ramdon_graph = Button(self.create_frame, text='GRAFO ALEATORIO', command=self.random_graph)
        # self.add_ramdon_graph.grid(row=4, column=0)

        # self.label1 = Label(self.create_frame, text='PERSONALIZADO', font='Calibri 10 bold underline')
        # self.label1.grid(row=0, column=0)

        self.add_vertices_button = Button(self.create_frame, text='AÑADIR NODOS', command=self.add_vertices_mode)
        self.add_vertices_button.grid(row=1, column=0)
        self.add_ribs_button = Button(self.create_frame, text='AÑADIR ARCOS', command=self.add_ribs_mode)
        self.add_ribs_button.grid(row=1, column=1)
        self.cancel_button = Button(self.create_frame, text='DESHACER', state=DISABLED, command=self.cancel)
        self.cancel_button.grid(row=1, column=2)
        self.complete_button = Button(self.create_frame, text='COMPLETAR', command=self.finish_function)
        self.complete_button.grid(row=1, column=3)

        # self.complete_button = Button(self.create_frame, text='CONFIGURACIÓN NODOS Y ARCOS', command=self.edit_color)
        # self.complete_button.grid(row=1, column=4)

    def _make_vertex_popup(self):
        self.vertex_popup_menu = Menu(self.root, tearoff=0)
        self.vertex_popup_menu.add_command(label='BORRAR NODO', command=lambda: self.popup_deleting(self.get_vertex,
                                                                                                    self.delete_vertex_by_name))
        self.vertex_popup_menu.add_command(label='RENOMBRAR', command=lambda: self.change_vertex_by_popup(StringVar(),
                                                                                                          self.rename_complete))  # self.rename_vertex)
        # self.vertex_popup_menu.add_command(label = 'CAMBIAR TAMAÑO', command = lambda: self.change_vertex_by_popup(IntVar(), self.resize_complete, self.g.ver_sizes))
        # self.vertex_popup_menu.add_command(label='CAMBIAR COLOR NODO', command = lambda: self.item_color_configure(self.get_vertex, self.g.ver_colours, 0))
        # self.vertex_popup_menu.add_command(label='CAMBIAR COLOR CONTORNO', command = lambda: self.item_color_configure(self.get_vertex, self.g.ver_colours, 1))

    def _make_rib_popup(self):
        self.rib_popup_menu = Menu(self.root, tearoff=0)
        self.rib_popup_menu.add_command(label='BORRAR ARCO',
                                        command=lambda: self.popup_deleting(self.get_rib, self.delete_rib_by_id))
        self.rib_popup_menu.add_command(label='CAMBIAR PESO', command=self.reweigh_rib_by_event)
        # self.rib_popup_menu.add_command(label='CAMBIAR COLOR ARCO', command = lambda: self.item_color_configure(self.get_rib, self.g.rib_colours))
        # self.rib_popup_menu.add_command(label='weight colour', command = lambda: self.item_color_configure(self.get_rib, self.g.weight_colours))

    def view_popup(self, ev, popup_menu):
        self.event = ev
        popup_menu.post(ev.x_root, ev.y_root)

    def clear_all(self):

       try:
        self.hide()
        self.history.append("Usuario borró el grafo - Hora: " + self.date());
        self.g = None
        self.canvas.delete('all')
       except:
           print("No hay nada que borrar")


    def set_color(self, variable, button):
        col = askcolor()
        if col[1]:
            variable.set(col[1])
            button.configure(bg=variable.get())

    def item_color_configure(self, find_function, graph_colours_dict, index=-1):
        n = find_function(self.event)
        self.event = None
        col = askcolor()[1]
        if col:
            if index != -1:
                cort = list(graph_colours_dict.get(n))
                cort[index] = col
                graph_colours_dict[n] = tuple(cort)
                self.canvas.view_vertex(self.g, n)
            else:
                graph_colours_dict[n] = col
                self.canvas.view_rib(self.g, n)
                self.canvas.view_rib_weight(self.g, n)

    def find_vertex(self, ev, size=0):  #####
        if not size:
            size = self.default_v_size.get()
        n = 0
        for name, coor in self.g.form.items():
            x, y = coor[0], coor[1]
            if x - size <= ev.x <= x + size and y - size <= ev.y <= y + size:
                n = name
                break
        return n

    def get_vertex(self, ev):
        tag = self.canvas.gettags(self.canvas.find_closest(ev.x, ev.y))[0]
        if 'rib' in tag or 'weight' in tag:
            return 0
        return tag

    def get_rib(self, ev):
        tag = self.canvas.gettags(self.canvas.find_closest(ev.x, ev.y))[0]
        # tag = self.canvas.gettags(ev.widget.find_withtag("current"))[0]
        i = int(tag[3:])
        return i

    def create_graph(self):
        self.canvas.delete('all')
        self.g = DictGraph()
        self.g.directed, self.g.weighted, self.g.multigraph = self.directed.get(), self.weighted.get(), self.multigraph.get()

        self.create_frame.grid(row=1, column=1, sticky='wn')

        # self._make_properties_frame()  # CAMBIO

        self._make_configure_frame()  # CAMBIO

        self.next_name = (name for name in self.names)
        self.add_vertices_mode()

    def m_create_graph(self):
        self.history.append("Usuario creó grafo personalizado - Hora: " + self.date());

    def edit_color(self):
        self._make_configure_frame()

    def edit_graph(self):


        if self.g:

            self.history.append("Usuario editó el grafo - Hora: " + self.date());

            self._make_configure_frame()

            self.create_frame.grid(row=1, column=1, sticky='wn')

            self.next_name = (name for name in [e for e in self.names if e not in self.g.vertices.keys()])
            self.add_vertices_mode()
        else:
            alert = Toplevel()
            message = Message(alert, width=200,
                              text='No hay ningun grafo abierto')
            button = Button(alert, text='OK',  bg='black', fg='white', font='sans 8 bold', command=lambda: alert.destroy())
            message.grid(row=0, column=0)
            button.grid(row=1, column=0)
            alert.bind('<Return>', lambda ev: alert.destroy())
            alert.focus()

    def add_vertices_mode(self):
        self.add_vertices_button.configure(relief=SUNKEN)
        self.add_ribs_button.configure(relief=RAISED)
        self.canvas.bind('<Button-1>', self.set_vertex)

        for e in self.g.vertices.keys():
            self.vertex_bindings(e)
            self.canvas.tag_bind(e, '<Button-1>', lambda ev: None)

    def add_ribs_mode(self):
        self.add_vertices_button.configure(relief=RAISED)
        self.add_ribs_button.configure(relief=SUNKEN)
        self.next = None
        self.canvas.bind('<Button-1>', lambda ev: None)

        for e in self.g.vertices.keys():
            self.canvas.tag_bind(e, '<B1-Motion>', self.set_rib)
            self.canvas.tag_bind(e, '<Button-1>', self.set_rib)

    def cancel(self):
        if self.cancel_stack:
            el = self.cancel_stack.pop()
            if type(el) is int:
                self.delete_rib_by_id(el)
            if type(el) is str:
                self.delete_vertex_by_name(el)
        if not self.cancel_stack:
            self.cancel_button.configure(state=DISABLED)

    def set_vertex(self, ev):
        if not self.find_vertex(ev):
            try:
                name = next(self.next_name)
                self.g.set_vertices(name)
                x = ev.x
                y = ev.y
                self.g.form[name] = (x, y)
                self.g.ver_sizes[name] = self.default_v_size.get()
                self.g.ver_colours[name] = (self.default_col_v_fill.get(), self.default_col_v_outline.get())
                self.canvas.view_vertex(self.g, name)
                self.vertex_bindings(name)
                if not self.cancel_stack:
                    self.cancel_button.configure(state=NORMAL)
                self.cancel_stack.append(name)
            except:
                '''
                alert = Toplevel()
                message = Message(alert, text = '', width = 400)
                button = Button(alert, text = '',  command = lambda: alert.destroy())
                message.grid(row=0, column=0)
                button.grid(row=1, column=0)
                alert.bind('<Return>', lambda ev: alert.destroy())
                alert.focus()
                '''

    def set_rib(self, ev):
        n = self.find_vertex(ev)
        if n:
            if self.next and self.next != n:
                if self.g.multigraph or self.g.get_ribs(self.next, n) == []:
                    dir = set()
                    if self.g.directed:
                        dir = {n}
                    i = self.g.set_rib(self.next, n, dir=dir)  # w ???
                    self.g.rib_colours[i] = self.default_col_r.get()
                    self.g.weight_colours[i] = self.default_col_weight.get()
                    self.canvas.view_rib(self.g, i)
                    if self.g.weighted:
                        self.canvas.view_rib_weight(self.g, i)

                    self.rib_bindings(i)
                    if not self.cancel_stack:
                        self.cancel_button.configure(state=NORMAL)
                    self.cancel_stack.append(i)
                    self.next = None
            else:
                self.next = n

    def finish_function(self):
        self.create_frame.grid_remove()

        #self.properties_frame.grid_remove()  # CAMBIO
        self.configure_frame.grid_remove()  # CAMBIO

        self.cancel_stack = []
        self.cancel_button.configure(state=DISABLED)
        self.canvas.bind('<Button-1>', lambda ev: None)
        for e in self.g.vertices.keys():
            self.vertex_bindings(e)
            self.canvas.tag_bind(e, '<Button-1>', lambda ev: None)
        self.next_name = None

    def name_check(self):
        if self.next_name:
            self.next_name = (name for name in [e for e in self.names if e not in self.g.vertices.keys()])

    def random_graph(self):
        # 1
        self.create_graph()

        self.ask = Toplevel()
        self.ask_vers.set(random.choice(range(4, 8)))
        if self.multigraph.get():
            self.ask_ribs.set(random.choice(range(8, 12)))
        else:
            self.ask_ribs.set(random.choice(range(4, 8)))

        Label(self.ask, text='NODOS').grid(row=0, column=0)
        Label(self.ask, text='ARCOS').grid(row=1, column=0)
        ver = Entry(self.ask, textvariable=self.ask_vers)
        ver.grid(row=0, column=1)

        rib = Entry(self.ask, textvariable=self.ask_ribs)
        rib.grid(row=1, column=1)

        ok_b = Button(self.ask, text='ACEPTAR', bg='black', fg='white', font='sans 8 bold',command=self.start_command)
        ok_b.grid(row=3, column=0, columnspan=2)

        self.ask.bind('<Return>', self.start_command)
        self.ask.focus()

        self.history.append("Usuario creó grafo aleatorio - Hora: " + self.date());

        self.hide()

    def start_command(self):
        self.g = random_graph(self.ask_vers.get(), self.ask_ribs.get(), weighted=self.weighted.get(),
                              directed=self.directed.get(),
                              multigraph=self.multigraph.get())

        c1 = '#000000'
        c2 = '#FF0040'
        self.default_col_v_fill.set(c1)
        self.default_col_v_outline.set(c2)

        self.default_col_weight.set(c1)
        self.default_col_r.set(c2)

        self.vertex_color_button.configure(bg=self.default_col_v_fill.get())
        self.outline_color_button.configure(bg=self.default_col_v_outline.get())
        self.rib_color_button.configure(bg=self.default_col_r.get())
        self.weight_color_button.configure(bg=self.default_col_weight.get())

        self.g.ver_colours = {e: (c1, c2) for e in self.g.vertices.keys()}
        self.g.rib_colours = {e: c2 for e in self.g.ribs.keys()}
        self.g.weight_colours = {e: c1 for e in self.g.ribs.keys()}

        self.canvas.view_graph(self.g)
        for e in self.g.vertices.keys():
            self.vertex_bindings(e)
        for i in self.g.ribs.keys():
            self.rib_bindings(i)

        self.ask.destroy()

    def vertex_bindings(self, ver):
        self.canvas.tag_bind(ver, '<Button-3>', lambda ev: self.view_popup(ev, self.vertex_popup_menu))
        self.canvas.tag_bind(ver, '<B1-Motion>', self.move_vertex_start)

    def rib_bindings(self, i):
        self.canvas.tag_bind('rib' + str(i), '<Button-3>', lambda ev: self.view_popup(ev, self.rib_popup_menu))
        self.canvas.tag_bind('weight' + str(i), '<Double-Button-1>', self.reweigh_rib_by_event)

    def saveimage(self):
        try:
            filename = asksaveasfilename()

            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            print("save")
            ImageGrab.grab().crop((x, y, x1, y1)).save(filename + ".png")

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** ¡Imagen generada exitosamente! ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

            v2.mainloop()

        except:

            self.history.append("Usuario exportó grafo en imagen - Hora: " + self.date());

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** Error al exportar imagen, intente nuevamente ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold',command=v2.destroy).pack()

            v2.mainloop()

    def savepdf(self):
        try:
            filename = asksaveasfilename()

            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            x1 = x + self.canvas.winfo_width()
            y1 = y + self.canvas.winfo_height()
            print("save")
            ImageGrab.grab().crop((x, y, x1, y1)).save(filename + ".pdf")

            self.history.append("Usuario exportó grafo en PDF - Hora: " + self.date());

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** ¡PDF generado exitosamente! ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

            v2.mainloop()

        except:

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** Error al exportar PDF, intente nuevamente ***").pack()
            Button(v2, text="OK",bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

            v2.mainloop()

    def date(self):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        return dt_string

    def savegrafo(self):

        # filename = asksaveasfilename()
        # f = open(filename, 'wb')
        # pickle.dump(self.g, f)

        #######

        try:
            # client = MongoClient('localhost', port=27017)
            client = MongoClient(
                "mongodb://test123:test123@cluster0-shard-00-00.brpwd.mongodb.net:27017,cluster0-shard-00-01.brpwd.mongodb.net:27017,cluster0-shard-00-02.brpwd.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-oaj24o-shard-0&authSource=admin&retryWrites=true&w=majority")

            db = client['grafos']
            col = db['grafo']

            mydoc = col.count_documents({})
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            grafo = list(self.g.ribs.values())
            nodos = list(self.g.vertices.keys())

            result = {
                'id_grafo': mydoc,
                'fecha': dt_string,
                'nodos': [i[0] for i in zip(nodos)],
                'grafo': [{'origen': i[0][0], 'destino': i[0][1], 'peso': i[0][2], 'sentido': i[0][1]} for i in
                          zip(grafo)],
                'dirigido': self.g.directed,
                'ponderado': self.g.weighted,
                'multigrafo': self.g.multigraph

            }

            test_string = json.dumps(result)
            res = ast.literal_eval(test_string)

            col.insert_one(res)

            print("Graph saved on DB!")

            self.history.append("Usuario guardó grafo en BD - Hora: " + self.date());

            v = Tk()

            Label(v, text="*** ¡Grafo almacenado exitosamente! (MongoDB Atlas) ***").pack()
            Button(v, text="OK", bg='black', fg='white', font='sans 8 bold',command=v.destroy).pack()

            v.mainloop()



        except:

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** Error al guardar en BD, grafo vacío ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold',command=v2.destroy).pack()

            v2.mainloop()

    def abrir(self):
        try:

            filename = askopenfilename()
            self.create_graph()
            f = open(filename, 'rb')
            self.g = pickle.load(f)
            # self.g.view()

            # pprint(vars(self.g))

            self.canvas.view_graph(self.g)



        except:
            pass

    def delete_rib_by_id(self, i):
        self.g.del_rib(i)
        self.canvas.delete_rib(i)

    def delete_vertex_by_name(self, name):
        self.canvas.delete_ribs(self.g, name)
        self.canvas.delete_vertex(name)
        self.g.del_vertex(name)
        self.name_check()

    def popup_deleting(self, find_function, delete_function):
        n = find_function(self.event)
        delete_function(n)
        self.event = None

    def move_vertex_start(self, ev):
        if self.next:
            self.canvas.bind('<ButtonRelease-1>', self.move_vertex_stop)  ### ??
        else:
            self.next = self.get_vertex(ev)

    def move_vertex_stop(self, ev):
        if self.next and not self.find_vertex(ev, self.g.ver_sizes.get(self.next)):
            ver = self.next
            self.g.form[ver] = (ev.x, ev.y)
            self.canvas.delete_ribs(self.g, ver)
            self.canvas.delete_vertex(ver)
            ribs = self.g.vertices.get(ver)
            for i in ribs:
                self.g.find_points_for_rib(i)
                self.canvas.view_rib(self.g, i)
                self.g.find_text_layout(i)
            if self.g.weighted:
                for i in ribs:
                    self.canvas.view_rib_weight(self.g, i)
            self.canvas.view_vertex(self.g, ver)
            self.next = None
            self.canvas.unbind('<ButtonRelease-1>')

    def change_vertex_by_popup(self, tvar, func, dict=None):
        if self.entry:
            self.entry.destroy()
        ev = self.event
        ver = self.get_vertex(ev)
        if dict:
            tvar.set(dict.get(ver))
        else:
            tvar.set(ver)
        self.entry = Entry(self.canvas, width=4, textvariable=tvar)
        self.entry.grid()
        self.entry.focus()
        x, y = self.g.form.get(ver)[0], self.g.form.get(ver)[1]
        self.canvas.create_window((x, y), window=self.entry)
        self.root.bind('<Return>', lambda ev: func(tvar, ver))
        self.root.bind('<Escape>', lambda ev: self.entry.destroy())  ##keybind update

    def rename_complete(self, tvar, old):
        new = tvar.get()
        if new not in self.g.vertices.keys() and new != '':
            self.rename_vertex(old, new)
            self.entry.destroy()
        if new == old:
            self.entry.destroy()

    def resize_complete(self, tvar, name):
        size = tvar.get()
        if size >= 10 and size <= 100:
            self.resize_vertex(name, size)
            self.entry.destroy()

    def reweigh_rib_by_event(self, ev=0):
        if self.entry:
            self.entry.destroy()
        if not ev:
            ev = self.event
        i = self.get_rib(ev)
        tvar = IntVar()
        tvar.set(self.g.ribs.get(i)[2])
        self.entry = Entry(self.canvas, width=4, textvariable=tvar)
        self.entry.grid()
        self.entry.focus()
        if self.g.multigraph:
            layout = self.g.rib_text_layout.get(i)[self.g.rib_orientation.get(i)][1]
        else:
            layout = self.g.rib_text_layout_simple.get(i)[self.g.rib_orientation.get(i)][1]
        x, y = layout[0], layout[1]
        self.canvas.create_window((x, y), window=self.entry)
        self.root.bind('<Return>', lambda ev: self.reweigh_complete(tvar, i))
        self.root.bind('<Escape>', lambda ev: self.entry.destroy())

    def reweigh_complete(self, tvar, i):
        weight = tvar.get()
        self.reweigh_rib(i, weight)
        self.entry.destroy()

    def rename_vertex(self, old, new):
        if new not in self.g.vertices.keys():
            self.g.rename_vertex(old, new)
            self.canvas.delete_vertex(old)
            self.canvas.view_vertex(self.g, new)
            self.canvas.tag_bind(new, '<Button-3>', lambda ev: self.view_popup(ev, self.vertex_popup_menu))
            self.canvas.tag_bind(new, '<B1-Motion>', self.move_vertex_start)
            self.name_check()

    def resize_vertex(self, name, size):
        self.g.ver_sizes[name] = size
        self.canvas.delete_vertex(name)
        self.canvas.view_vertex(self.g, name)
        self.canvas.delete_ribs(self.g, name)
        for rib in self.g.vertices.get(name):
            self.g.find_points_for_rib(rib)
            self.g.find_text_layout(rib)
            self.canvas.view_rib(self.g, rib)
            if self.g.weighted:
                self.canvas.view_rib_weight(self.g, rib)
        self.canvas.tag_bind(name, '<Button-3>', lambda ev: self.view_popup(ev, self.vertex_popup_menu))
        self.canvas.tag_bind(name, '<B1-Motion>', self.move_vertex_start)

    def reweigh_rib(self, i, new):
        rib = self.g.ribs.get(i)
        rib[2] = new
        self.canvas.delete_rib_weight(i)
        if self.g.weighted:
            self.canvas.view_rib_weight(self.g, i)

    ##########

    def serialize_sets(obj):
        if isinstance(obj, set):
            return list(obj)

        return obj

    def saveexcel(self):
        try:
            grafo = list(self.g.ribs.values())
            nodos = list(self.g.vertices.keys())

            origen = []
            destino = []
            peso = []
            sentido = []

            for i in grafo:
                origen.append(i[0])
                destino.append(i[1])
                peso.append(i[2])
                sentido.append(i[3])

            excel1 = pandas.DataFrame({
                'Origen': origen,
                'Destino': destino,
                'Peso': peso,
                'Sentido': sentido,
            })

            excel2 = pandas.DataFrame({
                'Nodos': nodos
            })

            '''excel3 = pandas.DataFrame({
                'Dirigido': list(self.g.directed),
                'Ponderado': list(self.g.weighted),
                'Multigrafo': list(self.g.multigraph)
            })'''

            new = pandas.concat([excel1, excel2], axis=1)

            f = asksaveasfilename()

            new.to_excel(str(f) + '.xlsx', sheet_name='sheet1', index=False)

            self.history.append("Usuario exportó grafo en excel - Hora: " + self.date());

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** ¡Excel generado exitosamente! ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

            v2.mainloop()

        except:
            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** Error al exportar, grafo vacío ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold',command=v2.destroy).pack()

            v2.mainloop()

    def guardarjson(self):
        try:

            grafo = list(self.g.ribs.values())
            nodos = list(self.g.vertices.keys())

            g = str(grafo)

            g = g.replace("{", "").replace("}", "")

            grafo = ast.literal_eval(g)


            result = {
                'nodos': [i[0] for i in zip(nodos)],
                'grafo': [{'origen': i[0][0], 'destino': i[0][1], 'peso': i[0][2], 'sentido': i[0][3]} for i in
                          zip(grafo)],
                'dirigido': self.g.directed,
                'ponderado': self.g.weighted,
                'multigrafo': self.g.multigraph,
            }

            json_string = json.dumps(result)
            # print(json_string)

            # f='C:/Users/alejo/OneDrive/Escritorio/a.json'

            f = asksaveasfilename()

            with open(str(f) + '.json', 'w') as outfile:
                outfile.write(json_string)

            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** ¡JSON guardado exitosamente! ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

            v2.mainloop()

        except:
            v2 = Tk()

            v2.eval('tk::PlaceWindow . center')

            Label(v2, text="*** Error al guardar JSON, grafo vacío ***").pack()
            Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold',command=v2.destroy).pack()

            v2.mainloop()

    def importar(self):
      try:
        self.create_graph()
        f = askopenfilename()
        nodos = []
        grafo = []
        # f = open('C:/Users/alejo/OneDrive/Escritorio/a.json')
        f = open(str(f))

        data = json.loads(f.read())

        print(data)

        for i in data['nodos']:
            nodos.append(i)

        for i in data['grafo']:
            grafo.append(list(i.values()))

        self.g.weighted = data['ponderado']
        self.g.directed = data['dirigido']
        self.g.multigraph == data['multigrafo']

        p = int(data['ponderado'])

        d = int(data['dirigido'])

        m = int(data['multigrafo'])

        f.close()

        self.s(nodos, grafo, p, d, m)

        self.hide()

        self.history.append("Usuario importó el grafo desde JSON - Hora: " + self.date());

        v2 = Tk()

        v2.eval('tk::PlaceWindow . center')

        Label(v2, text="*** ¡JSON importado exitosamente! ***").pack()
        Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

        v2.mainloop()

      except:
          v2 = Tk()

          v2.eval('tk::PlaceWindow . center')

          Label(v2, text="*** Error al abrir JSON, intente nuevamente ***").pack()
          Button(v2, text="OK", bg='black', fg='white', font='sans 8 bold', command=v2.destroy).pack()

          v2.mainloop()

    def s(self, nodos, grafo, p, d, m):
        self.g = imp(nodos, grafo, p, d, m)

        c1 = '#000000'
        c2 = '#FF0040'
        self.default_col_v_fill.set(c1)
        self.default_col_v_outline.set(c2)

        self.default_col_weight.set(c1)
        self.default_col_r.set(c2)

        self.vertex_color_button.configure(bg=self.default_col_v_fill.get())
        self.outline_color_button.configure(bg=self.default_col_v_outline.get())
        self.rib_color_button.configure(bg=self.default_col_r.get())
        self.weight_color_button.configure(bg=self.default_col_weight.get())

        self.g.ver_colours = {e: (c1, c2) for e in self.g.vertices.keys()}
        self.g.rib_colours = {e: c2 for e in self.g.ribs.keys()}
        self.g.weight_colours = {e: c1 for e in self.g.ribs.keys()}

        self.canvas.view_graph(self.g)
        for e in self.g.vertices.keys():
            self.vertex_bindings(e)
        for i in self.g.ribs.keys():
            self.rib_bindings(i)

    def imprimir2(self):

        filename = asksaveasfilename()

        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        print("save")
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename + ".pdf")

        os.startfile(filename + ".pdf", "print")

        self.history.append("Usuario imprimió el grafo - Hora: " + self.date());

    def imprimir(self):

        filename = asksaveasfilename()

        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        print("save")
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename + ".pdf")

        currentprinter = win32print.GetDefaultPrinter()
        win32api.ShellExecute(0, "print", str(filename) + ".pdf", '/d:"%s"' % currentprinter, ".", 0)

    def mostrarDatos(self, tabla):

        MONGO_HOST = "localhost"
        MONGO_PUERTO = "27017"
        MONGO_TIEMPO_FUERA = 1000
        MONGO_URI = "mongodb://" + MONGO_HOST + ":" + MONGO_PUERTO + "/"
        MONGO_BASEDATOS = "grafos"
        MONGO_COLECCION = "grafo"

        try:
            # cliente = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=MONGO_TIEMPO_FUERA)

            cliente = MongoClient(
                "mongodb://test123:test123@cluster0-shard-00-00.brpwd.mongodb.net:27017,cluster0-shard-00-01.brpwd.mongodb.net:27017,cluster0-shard-00-02.brpwd.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-oaj24o-shard-0&authSource=admin&retryWrites=true&w=majority")

            baseDatos = cliente[MONGO_BASEDATOS]
            coleccion = baseDatos[MONGO_COLECCION]

            for documento in coleccion.find():
                aux = (','.join(documento["nodos"]))

                tabla.insert('', 0, text=documento["id_grafo"], values=(aux, documento["fecha"]))

            cliente.close()

        except pymongo.errors.ServerSelectionTimeoutError as errorTiempo:
            print("Tiempo exedido " + errorTiempo)
        except pymongo.errors.ConnectionFailure as errorConexion:
            print("Fallo al conectarse a mongodb " + errorConexion)

    def mongo(self):

        ventana = Tk()
        ventana.title("MongoDB Atlas ")
        tabla = ttk.Treeview(ventana, columns=('#0', '#1'))
        tabla.grid(row=1, column=0, columnspan=1)
        tabla.heading("#0", text="ID")
        tabla.heading("#1", text="NODOS")
        tabla.heading("#2", text="FECHA CREACIÓN")
        self.mostrarDatos(tabla)

        lbl = ttk.Label(ventana, text="INGRESE EL ID DEL GRAFO: ")
        lbl.grid(column=3, row=1)

        nameEntered = ttk.Entry(ventana)
        nameEntered.grid(column=4, row=1, sticky=W)

        button = ttk.Button(ventana, text="CARGAR", command=lambda: [self.prueba(nameEntered), ventana.destroy()])
        button.grid(column=5, row=1)

        ventana.mainloop()

        self.history.append("Usuario importó el grafo desde BD - Hora: " + self.date());

    def prueba(self, nameEntered):
        aux = int(nameEntered.get())

        # mongo_client = MongoClient('mongodb://localhost:27017')

        mongo_client = MongoClient(
            "mongodb://test123:test123@cluster0-shard-00-00.brpwd.mongodb.net:27017,cluster0-shard-00-01.brpwd.mongodb.net:27017,cluster0-shard-00-02.brpwd.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-oaj24o-shard-0&authSource=admin&retryWrites=true&w=majority")

        ll = list(mongo_client['grafos']['grafo'].find(filter={'id_grafo': aux}))

        l = dumps(ll)

        d = str(l)
        d1 = (d[1:])
        d2 = (d1[:-1])

        data = json.loads(d2)

        self.create_graph()

        nodos = []
        grafo = []

        for i in data['nodos']:
            nodos.append(i)

        for i in data['grafo']:
            grafo.append(list(i.values()))

        self.g.weighted = data['ponderado']
        self.g.directed = data['dirigido']
        self.g.multigraph == data['multigrafo']

        p = int(data['ponderado'])

        d = int(data['dirigido'])

        m = int(data['multigrafo'])

        self.s(nodos, grafo, p, d, m)

        self.hide()


    def eliminar_db(self,nameEntered):
        aux = int(nameEntered.get())


        mongo_client = MongoClient(
            "mongodb://test123:test123@cluster0-shard-00-00.brpwd.mongodb.net:27017,cluster0-shard-00-01.brpwd.mongodb.net:27017,cluster0-shard-00-02.brpwd.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-oaj24o-shard-0&authSource=admin&retryWrites=true&w=majority")

        m= mongo_client['grafos']['grafo']

        myquery = {"address": aux}

        m.delete_one(myquery)

    def inicio(self):

        class Table:

            def __init__(self, root):



                for j in range(total_columns):
                    self.e = Entry(root, width=70, font=('Calibri', 12, 'bold'))

                    self.e.grid(row=j, column=1)

                    self.e.insert(END, lst[j])

                b = Button(root, text="OK", bg='black', fg='white', font='sans 8 bold',command=root.destroy)
                b.grid(row=len(lst), column=1)

        lst = self.history
        total_columns = len(lst)

        root = Tk()
        root.eval('tk::PlaceWindow . center')
        root.title("** Ejecuciones realizadas **")

        t = Table(root)

        root.mainloop()

    def hide(self):
        self.create_frame.grid_remove()

        self.configure_frame.grid_remove()


    def website(self):
        webbrowser.open("https://drive.google.com/file/d/1btaYYO503juqUd2pNbzHU6mW-bgT2cK5/view")

    def website2(self):
        webbrowser.open("https://drive.google.com/file/d/1VSuKIcUNscanoi_tlg2VmRmSxk99pqFj/view?usp=drive_open")

    def pp(self):
        grafo = list(self.g.ribs.values())

        g = str(grafo)

        g = g.replace("{", "").replace("}", "")

        grafo = ast.literal_eval(g)

        origen = [i[0][0] for i in zip(grafo)]

        destino= [i[0][1] for i in zip(grafo)]

        peso = [i[0][2] for i in zip(grafo)]

        sentido = [i[0][3] for i in zip(grafo)]


        window = Tk()

        treev = ttk.Treeview(window, selectmode='browse')
        treev.pack(side='left', expand=True, fill='both')

        verscrlbar = ttk.Scrollbar(window,
                                   orient="vertical",
                                   command=treev.yview)

        verscrlbar.pack(side='right', fill='y')
        treev.configure(yscrollcommand=verscrlbar.set)

        treev["columns"] = ('1', '2','3','4')

        treev['show'] = 'headings'

        treev.column("1", width=90, anchor='c')
        treev.column("2", width=90, anchor='c')
        treev.column("3", width=90, anchor='c')
        treev.column("4", width=90, anchor='c')

        treev.heading("1", text="Origen")
        treev.heading("2", text="Destino")
        treev.heading("3", text="Peso")
        treev.heading("4", text="Sentido")

        for x, y, a, b in zip(origen, destino, peso, sentido):
            treev.insert("", 'end', values=(x, y,a,b))

        window.mainloop()





    def pp2(self):


        edges=[*self.g.ribs.values()]
        nodes=[*self.g.vertices.keys()]

        edgesT = [tuple(x) for x in edges]

        edgesF = [tuple(s if s != s else self.letter_to_int(s) for s in tup) for tup in edgesT]

        nodesF=[self.letter_to_int(i) if i==i else i for i in nodes]


        n_nodes = len(nodesF)
        A = np.zeros((n_nodes, n_nodes))

        for edge in edgesF:
            i = int(edge[0])
            j = int(edge[1])
            weight = edge[2]
            A[i, j] = weight
            A[j, i] = weight


        print(A)



    def letter_to_int(self, letter):
        try:
            alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            return alphabet.index(letter)
        except:
            return letter


'''
def random_color():
    l = ['00','11','22','33','44','55','66','77','88','99','aa','bb','cc','dd','ee','ff']
    cs = '#'+'00'*3
    while cs == '#000000' or cs == '#ffffff':
        cs = [random.choice(l),random.choice(l),random.choice(l)]
        cs = '#'+cs[0]+cs[1]+cs[2]


    return cs
'''

if __name__ == '__main__':
    root = Window()

from pymongo import MongoClient, ASCENDING, DESCENDING

# create a client instance of MongoClient
mongo_client = MongoClient('mongodb://localhost:27017')




from tkinter import*
from tkinter import ttk
from tkinter import messagebox
import pymongo
MONGO_HOST="localhost"
MONGO_PUERTO="27017"
MONGO_TIEMPO_FUERA=1000
MONGO_URI="mongodb://"+MONGO_HOST+":"+MONGO_PUERTO+"/"
MONGO_BASEDATOS="grafos"
MONGO_COLECCION="grafo"

def mostrarDatos(tabla):
    try:
        cliente=pymongo.MongoClient(MONGO_URI,serverSelectionTimeoutMS=MONGO_TIEMPO_FUERA)
        baseDatos=cliente[MONGO_BASEDATOS]
        coleccion=baseDatos[MONGO_COLECCION]
        for documento in coleccion.find():


            aux=(','.join(documento["nodos"]))

            tabla.insert('', 0, text=documento["id_grafo"], values=(aux,documento["fecha"]))

        cliente.close()

    except pymongo.errors.ServerSelectionTimeoutError as errorTiempo:
        print("Tiempo exedido "+errorTiempo)
    except pymongo.errors.ConnectionFailure as errorConexion:
        print("Fallo al conectarse a mongodb "+errorConexion)


    result = mongo_client['grafos']['grafo'].find(filter={'id_grafo': 1})

    print(list(result))
    print("prueba")


ventana=Tk()
tabla=ttk.Treeview(ventana,columns = ('#0','#1','#2'))
tabla.grid(row=1,column=0,columnspan=1)
tabla.heading("#0",text="ID")
tabla.heading("#1",text="NODOS")
tabla.heading("#2",text="FECHA CREACIÓN")
mostrarDatos(tabla)
ventana.mainloop()





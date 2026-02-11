import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import pandas as pd
from scipy.constants import c,k,h   


# definizione costanti utili


n=1.00029# Indice di rifrazione dell'atmosfera terrestre
N=2.504e25# Densità di molecole dell'atmosfera[mol*m^-3]
T_s=5.75e3# Temperatura del Sole[K]
T_a=3.7e3# Temperatura di Antares[K]
T_v=10e3# Temperatura di Vega[K]
T_r=25e3# Temperatura di Rigel[K]
R_t=6371000# Raggio della Terra [m]
S_z=8000# Spessore massa d'aria allo zenith [m]
S_oriz=np.sqrt(np.power(R_t+S_z, 2)-np.power(R_t, 2))# m
L_tot=np.linspace(10e-9,3000e-9,10000)# Seleziona 10000 lunghezze d'onda tra 10 e 3000 nm


#definizione delle funzioni



defB(L, T):
    """
    Funzione che descrive la densità di energia irradiata
    da un corpo di temperatura T in funzione 
    della lunghezza d'onda lambda
    
    Parametri:
        L : Lunghezza d'onda della radiazione EM emessa dal corpo [m]
        T : Temperatura del corpo [K]
    
    Restituisce (2hc^2/L^5)*(1/e^(hc/L*k*T))-1)[J*m^-3*s^-1], con h costante di Planck,
    k costante di Boltzmann e c velocità della luce nel vuoto
    """
    expo=h*c/(L*k*T)
    return (2*h*np.power(c, 2)/np.power(L, 5))*(1/(np.exp(expo)-1))
def E_L(L):
    """"
    Funzione che esprime l'energia dei fotoni
    in funzione della lunghezza d'onda
    Parametri:
        L : Lunghezza d'onda dei fotoni [m]
    
    Restituisce h*c/L
    """
    return (h*c/L)
def E_nu(nu):
    """
    Funzione che esprime l'energia dei fotoni
    in funzione della frequenza
    Parametri:
        nu : Frequenza dei fotoni [s^-1]
    
    Restituisce h*nu
    """
    return h*nu
def D(L, T):
    """
    Funzione che descrive la densità di fotoni
    per lunghezza d'onda
    Parametri:
        L : Lunghezza d'onda [m]
        T : Temperatura[K]
    
    Restituisce B(L, T)/E, dove E=E_L [fotoni*m^-3*s^-1]
    """
    expo=(h*c)/(L*k*T)
    return (2*h*np.power(c, 2)/np.power(L, 5))*(1/(np.exp(expo)-1))*(L/(h*c))
def beta(L, n, Nc):
    """
    Funzione che descrive lo scattering di Rayleigh in funzione 
    della lunghezza d'onda, dell'indice di rifrazione e alla densità di molecole 
    Parametri:
        L : Lunghezza d'onda [m]
        n : Indice di rifrazione 
        Nc : Densità di molecole [molecole*m^-3]
    
    Restituisce (8*pi^3/3*L^4*N)*(n^2-1)^2[m^-1]
    """
    return (8*np.pi**3/(3*L**4*N))*(n**2-1)**2  
def N_0(L, T):
    """
    Funzione che descrive il numero di fotoni
    iniziali, prima che raggiungano l'atmosfera terrestre: si noti
    che coincide con la densità di fotoni per lunghezza d'onda 
    alla temperatura della Stella in esame
    Parametri:
        L : Lunghezza d'onda [m]
    
    Restituisce D(L, T_Stella)
    """
    return D(L, T)
def N_obs(L, S, T):
    """
    Funzione che descrive il numero di fotoni osservati
    ad una certa lunghezza d'onda senza essere stati deviati
    Parametri:
        L : Lunghezza d'onda [m]   
        S : Spessore massa d'aria, lunghezza del percorso in atmosfera [m]
        T : Temperatura della Stella [K]
    
    Restituisce N_0(L)*exp(-beta(L, n, N)*S)
    """
    expo=-beta(L, n, N)*S
    return N_0(L, T)*np.exp(expo)
def S_theta(th):
    """
    Funzione che approssima lo spessore della massa d'aria 
    considerando un qualsiasi angolo theta
    rispetto allo Zenith 
    Parametri:
        th : Angolo rispetto allo Zenith [rad]
    
    Restituisce sqrt((R_t*cos(th))^2+2*R_t*S_z+S_z^2)-R_t*cos(th)
    """
    return np.sqrt(np.power(R_t*np.cos(th), 2)+2*R_t*S_z+np.power(S_z, 2))-R_t*np.cos(th)
def hm(L, S, T, Ns):
    """
    Funzione che definisce il metodo hit or miss
    per una determinata lunghezza d'onda, il cammino dei fotoni,
    la temperatura della stella in esame e il numero di campioni
    Parametri:
        L : Array di lunghezze d'onda [m]
        S : Spessore massa d'aria [m]
        T : Temperatura della stella [K]
        Ns : numero di campioni scelto 
        
    Restituisce
    """
    if S==0:
        F=N_0(L, T)
    else:
        F=N_obs(L, S, T)
    Fmax=np.max(F)
    xhm=np.random.uniform(low=np.min(L),high=np.max(L),size=Ns)
    yhm=np.random.random(Ns)
    maskhm=yhm<=N_obs(xhm, S, T)/Fmax
    xnew=xhm[maskhm]
    return xnew
def flusso(L, th, Ns, T):
    """
    Funzione che calcola il flusso relativo di fotoni in funzione dell'angolo
    theta sfruttando il metodo della media
    Parametri:
        L : Lunghezza d'onda [m]
        th : Angolo che descrive la posizione del Sole rispetto allo Zenith [rad]
        Ns : Numero di fotoni che si vuole campionare
        T : Temperatura della Stella [K]
    Restituisce il flusso come (L_max-L_min)*media(fotoni)
    """
    if th==np.pi/2:
        S_m=S_oriz
    else:
        S_m=S_theta(th)    
    L_r=np.random.uniform(low=np.min(L),high=np.max(L),size=Ns)
    val=N_obs(L_r, S_m, T)
    return (np.max(L)-np.min(L))*np.mean(val)



#definizione del menù che gestisce la prima parte del progetto 



def menu_interattivo():
    """
    Funzione che gestisce il programma: entra in gioco con la selezione mediante ArgParse,
    successivamente fa partire un ciclo while che permette all'utente di scegliere i dati e visualizzare 
    la risposta del programma. I dati sono limitati a quelli presenti nella consegna del progetto
    """
    #definisco il ciclo while che gestisce il menù
    
    while True:
        print("\nSelezionare la stella")
        print("S: Sole")
        print("A: Antares")
        print("V: Vega")
        print("R: Rigel")
        print("Q: Uscire dal programma")
        scelta=input(">>>")
        if scelta == "Q":
            print("Fine esecuzione...")
            break
        stelle={"S":("Sole", T_s), "A":("Antares", T_a), "V":("Vega", T_v), "R":("Rigel", T_r)}
        if scelta not in stelle:
            print("Scelta non valida")
            continue
        nome,T=stelle[scelta]
        N_f=input("Inserire il numero di fotoni da campionare: ")
        N_fot=int(N_f)
        
        
        #definisco un ciclo while annidato per la selezione e lo scarto degli angoli
        
        
        while True:
            S=input("Inserire l'angolo in gradi per lo spessore della massa d'aria: ")
            S_fl=float(S)
            S_rad=((np.pi*S_fl)/180)
            if S_rad>np.pi/2 or S_rad<-(np.pi/2):
                print("Bisogna inserire un angolo tra -90° e 90°")
            else:
                break
        
        
        #chiamo le funzioni e definisco il caso limite
        
        
        hm1=hm(L_tot, 0, T, N_fot)
        hm2=hm(L_tot, S_z, T, N_fot)
        hm3=hm(L_tot, S_oriz, T, N_fot)
        if S_rad==np.pi/2:
            hm4=hm(L_tot, S_oriz, T, N_fot)
        else: 
            hm4=hm(L_tot, S_theta(S_rad), T, N_fot)
        
        
        #plotto i grafici delle 4 funzioni
        
        
        plt.figure(figsize=(12,8))
        plt.hist(hm1,bins=300,range=((np.min(L_tot)),np.max(L_tot)),color='tomato',alpha=0.8,label="senza assorbimento")
        plt.hist(hm2,bins=300,range=((np.min(L_tot)),np.max(L_tot)),color='gold',alpha=0.8,label="Zenith")
        plt.hist(hm3,bins=300,range=((np.min(L_tot)),np.max(L_tot)),color='g',alpha=0.8,label="Orizzonte")
        plt.hist(hm4,bins=300,range=((np.min(L_tot)),np.max(L_tot)),color='navy',alpha=0.8,label="Angolo scelto")
        plt.axvspan(380e-9, 800e-9,facecolor="#B0C4DE",edgecolor="black",alpha=0.25,linewidth=1.2)
        plt.text((380e-9+800e-9)/2,0.98,"Spettro visibile",ha="center",va="top",fontsize=11,fontstyle="italic",color="black",alpha=0.9,transform=plt.gca().get_xaxis_transform())
        plt.xlabel(r"$\lambda$[m]",fontstyle="italic")
        plt.ylabel("Conteggio",fontstyle="italic")
        plt.title("Distribuzione di fotoni simulata")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
        
        #rifaccio un altro ciclo while annidato per poter riscegliere un altro angolo, non necessariamente uguale al primo
        
        
        print("\nInserire l'angolo che descrive la posizione della Stella rispetto allo zenith (in gradi):")
        while True:
            ang=input(">>>")
            ang_fl=float(ang)
            ang_rad=((np.pi*ang_fl)/180)
            if ang_rad>np.pi/2 or ang_rad<-(np.pi/2):
                print("Bisogna inserire un angolo tra -90° e 90°")
            else:
                break
        flux=flusso(L_tot, ang_rad, N_fot, T)
        print("Flusso osservato di fotoni:[fot*s^-1*m^-2]", flux)


#definizione della funzione che si occupa dello studio sull'Ozono



def Stud_O3():
    """
    Funzione che gestisce la parte di studio qualitativo dell'assorbimento dell'Ozono. Partendo da un file 
    che mostra la cross-section in funzione della lunghezza d'onda, mostra un confronto tra le diverse temperature
    """
    
    
    #carico il file come dataframe e rinomino le colonne
    
    
    df=pd.read_csv("SCIA_O3_Temp_cross-section_V4.1.DAT",comment="!",sep=r"\s+",header=None)
    df.columns=["vacuum_wavelength","cross_section_203k","cross_section_223k","cross_section_243k","cross_section_273k","cross_section_293k"]
    
    
    #Creo la figura che contiene i grafici che mostrano i dati caricati
    
    
    fig,axs=plt.subplots(2,3,figsize=(16,8))
    axs[0,0].plot(df["vacuum_wavelength"],df["cross_section_203k"],color="indigo")
    axs[0,0].set_yscale("log")
    axs[0,0].set_xlabel(r"$\lambda$ [nm]")
    axs[0,0].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[0,0].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[0,0].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[0,0].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[0,0].set_title("Sezione d'urto dell'Ozono a 203K")
    axs[0,0].legend()
    axs[0,1].plot(df["vacuum_wavelength"],df["cross_section_223k"],color="darkturquoise")
    axs[0,1].set_yscale("log")
    axs[0,1].set_xlabel(r"$\lambda$ [nm]")
    axs[0,1].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[0,1].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[0,1].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[0,1].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[0,1].set_title("Sezione d'urto dell'Ozono a 223K")
    axs[0,1].legend()
    axs[0,2].plot(df["vacuum_wavelength"],df["cross_section_243k"],color="darkkhaki")
    axs[0,2].set_yscale("log")
    axs[0,2].set_xlabel(r"$\lambda$ [nm]")
    axs[0,2].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[0,2].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[0,2].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[0,2].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[0,2].set_title("Sezione d'urto dell'Ozono a 243K")
    axs[0,2].legend()
    axs[1,0].plot(df["vacuum_wavelength"],df["cross_section_273k"],color="crimson")
    axs[1,0].set_yscale("log")
    axs[1,0].set_xlabel(r"$\lambda$ [nm]")
    axs[1,0].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[1,0].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[1,0].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[1,0].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[1,0].set_title("Sezione d'urto dell'Ozono a 273K")
    axs[1,0].legend()
    axs[1,1].plot(df["vacuum_wavelength"],df["cross_section_293k"],color="darkslategray")
    axs[1,1].set_yscale("log")
    axs[1,1].set_xlabel(r"$\lambda$ [nm]")
    axs[1,1].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[1,1].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[1,1].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[1,1].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[1,1].set_title("Sezione d'urto dell'Ozono a 293K")
    axs[1,1].legend()
    axs[1,2].plot(df["vacuum_wavelength"],df["cross_section_203k"],color="indigo")
    axs[1,2].plot(df["vacuum_wavelength"],df["cross_section_223k"],color="darkturquoise")
    axs[1,2].plot(df["vacuum_wavelength"],df["cross_section_243k"],color="darkkhaki")
    axs[1,2].plot(df["vacuum_wavelength"],df["cross_section_273k"],color="crimson")
    axs[1,2].plot(df["vacuum_wavelength"],df["cross_section_293k"],color="darkslategray")
    axs[1,2].set_yscale("log")
    axs[1,2].set_xlabel(r"$\lambda$ [nm]")
    axs[1,2].set_ylabel(r"$\sigma(\lambda)[cm^{-2}$]")
    axs[1,2].axvspan(380,800,facecolor="white",alpha=0.15,edgecolor="black",linewidth=1.2,label="Spettro visibile")
    axs[1,2].axvspan(np.min(df["vacuum_wavelength"].values),380,color="purple",alpha=0.5,label="Zona UV")
    axs[1,2].axvspan(800,np.max(df["vacuum_wavelength"].values),color="darkred",alpha=0.5,label="Zona IR")
    axs[1,2].set_title("Grafico riassuntivo")
    axs[1,2].legend()
    plt.tight_layout()
    plt.show() 
    return 0

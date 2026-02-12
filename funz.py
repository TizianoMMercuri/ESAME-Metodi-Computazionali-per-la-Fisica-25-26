import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.constants import c,k,h   
from scipy.interpolate import interp1d

# definizione costanti utili


n=1.00029# Indice di rifrazione dell'atmosfera terrestre
N=2.504e25# Densità di molecole dell'atmosfera[mol*m^-3]
T_s=5.75e3# Temperatura del Sole[K]
T_a=3.7e3# Temperatura di Antares[K]
T_v=10e3# Temperatura di Vega[K]
T_r=25e3# Temperatura di Rigel[K]
R_t=6.371e6# Raggio della Terra [m]
S_z=8000# Spessore massa d'aria allo zenith [m]
S_oriz=np.sqrt(np.power(R_t+S_z, 2)-np.power(R_t, 2))# m
L_tot=np.linspace(10,3000,300)# Seleziona 10000 lunghezze d'onda tra 10 e 3000 nm
N_fot=500000# Numero dei fotoni che si vanno a campionare


#carico il file come dataframe e rinomino le colonne


df=pd.read_csv("SCIA_O3_Temp_cross-section_V4.1.DAT",comment="!",sep=r"\s+",header=None)
df.columns=["vacuum_wavelength","cross_section_203k","cross_section_223k","cross_section_243k","cross_section_273k","cross_section_293k"]


#definizione delle funzioni



def B(T):
    """
    Funzione che descrive la densità di energia irradiata
    da un corpo di temperatura T in funzione 
    della lunghezza d'onda lambda
    
    Parametri:
        T : Temperatura del corpo [K]
    
    Restituisce (2hc^2/L^5)*(1/e^(hc/L*k*T))-1)[J*m^-3*s^-1], con h costante di Planck,
    k costante di Boltzmann e c velocità della luce nel vuoto
    """
    expo=h*c/((L_tot*1e-9)*k*T)
    return (2*h*np.power(c, 2)/np.power((L_tot*1e-9), 5))*(1/(np.exp(expo)-1))
def D(L, T):
    """
    Funzione che descrive la densità di fotoni
    per lunghezza d'onda
    
    Parametri:
        L : Lunghezza d'onda [nm]
        T : Temperatura[K]
    
    Restituisce B(L, T)/E, dove E=E_L [fotoni*m^-3*s^-1]
    """
    L_m=L*1e-9
    expo=(h*c)/(L_m*k*T)
    return (2*c)/(L_m**4*(np.exp(expo)-1))
def beta():
    """
    Funzione che descrive lo scattering di Rayleigh in funzione 
    della lunghezza d'onda, dell'indice di rifrazione e alla densità di molecole 
    
    
    Restituisce (8*pi^3/(3*L^4*N))*(n^2-1)^2[m^-1]
    """
    return (8*np.pi**3/(3*(L_tot*1e-9)**4*N))*(n**2-1)**2  
def N_obs(S, D):
    """
    Funzione che descrive il numero di fotoni osservati
    ad una certa lunghezza d'onda senza essere stati deviati
    
    Parametri:   
        S : Spessore massa d'aria, lunghezza del percorso in atmosfera [m]
        D : Numero di conteggi restituiti dal metodo hit or miss
    
    Restituisce D*exp(-beta(L, n, N)*S)
    """
    expo=np.exp(-beta()*S)
    return D*expo
def S_theta(th):
    """
    Funzione che approssima lo spessore della massa d'aria 
    considerando un qualsiasi angolo theta
    rispetto allo Zenith 
    
    Parametri:
        th : Angolo rispetto allo Zenith [rad]
    
    Restituisce sqrt((R_t*cos(th))^2+2*R_t*S_z+S_z^2)-R_t*cos(th)
    """
    th_rad=((np.pi*th)/180)
    return np.sqrt(np.power(R_t*np.cos(th_rad), 2)+2*R_t*S_z+np.power(S_z, 2))-R_t*np.cos(th_rad)
def hm(T):
    """
    Funzione che definisce il metodo hit or miss
    
    Parametri:
        T : Temperatura della stella [K]
    
    Restituisce un conteggio di fotoni 
    """
    xhm=np.random.uniform(low=np.min(L_tot),high=np.max(L_tot),size=N_fot)
    yhm=np.random.random(N_fot)
    maskhm=yhm<=(D(xhm, T)/D((2898/T)*1000, T))
    xnew=xhm[maskhm]
    hist=np.histogram(xnew, bins=300, range=(10, 3000))
    return hist[0]
def zenith(sam):
    """
    Funzione che imposta il metodo hit or miss quando lo spessore d'aria coincide con lo zenith
    
    Parametri:
        sam : conteggio restituito da hm(T) 
    """
    zen=N_obs(S_z, sam)
    return zen
def oriz(sam):
    """
    Funzione che imposta il metodo hit or miss quando lo spessore d'aria coincide con l'orizzonte
    
    Parametri:
        sam : conteggio restituito da hm(T) 
    """
    oor=N_obs(S_oriz, sam)
    return oor
def phi(sam, th):
    """
    Funzione che imposta il metodo hit or miss quando lo spessore d'aria viene rappresentato
    da un angolo theta
    
    Parametri:
        sam : conteggio restituito da hm(T) 
        th : angolo [°]
    """
    s_phi=S_theta(th)
    thet=N_obs(s_phi, sam)
    return thet
def flusso(sam, th):
    """
    Funzione che calcola il flusso relativo di fotoni in funzione dell'angolo
    theta
    Parametri:
        sam : conteggio restituito da hm(T) 
        th : Angolo che descrive la posizione del Sole rispetto allo Zenith [°]
        
    Restituisce il flusso relativo
    """
    theta_th=np.exp(-beta()*S_theta(th))
    theta_zen=np.exp(-beta()*S_z)
    flusso_th=np.sum(sam*theta_th)
    flusso_zen=np.sum(sam*theta_zen)
    return flusso_th/flusso_zen
def assorb_O3(L_O3, S, du=300, T_O3=243):
    """
    Funzione che imposta l'assorbimento da parte dell'Ozono con i dati che si hanno a disposizione dal file caricato
    
    Parametri:
        L_O3 : Lunghezza d'onda per lo studio[nm]
        S : Spessore della massa d'aria[m]
        du : numero che definisce una densità efficace di molecole di O3 nell'atmosfera [DU]
        T_O3 : Temperatura efficace per l'O3 [K]
        
    Restituisce l'assorbimento da parte dell'Ozono 
    """
    f=interp1d(df["vacuum_wavelength"], df["cross_section_243k"], kind='linear',fill_value='extrapolate', bounds_error=False)
    sigma=f(L_O3)
    sigma_m2=sigma*1e-4
    col_O3=du*2.69e20
    col_th=col_O3*(S/S_z)
    return np.exp(-sigma_m2*col_O3)


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
        
        
        #definisco un ciclo while annidato per la selezione e lo scarto degli angoli
        
        
        while True:
            S=input("Inserire l'angolo per lo spessore della massa d'aria: ")
            S_fl=float(S)
            if S_fl>90 or S_fl<0:
                print("Bisogna inserire un angolo tra 0° e 90°")
            else:
                break
        
        
        #plotto i grafici delle 4 funzioni
        
        ph=hm(T)   
        z=zenith(ph)
        o=oriz(ph)
        f=phi(ph, S_fl)
        plt.figure(figsize=(12,8))
        plt.bar(L_tot, height=ph, width=(L_tot[1]-L_tot[0]), color='indigo', alpha=0.6, label="Nessun assorbimento",linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, height=z, width=(L_tot[1]-L_tot[0]), color='gold', alpha=0.6, label="Zenith",linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, height=f, width=(L_tot[1]-L_tot[0]), color='green', alpha=0.6, label="Angolo scelto",linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, height=o, width=(L_tot[1]-L_tot[0]), color='red', alpha=0.6, label="Orizzonte",linewidth=0.5, edgecolor='black')
        plt.xlabel(r"$\lambda$[m]",fontstyle="italic")
        plt.ylabel("Conteggio",fontstyle="italic")
        plt.title("Distribuzione di fotoni simulata")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        
        alpha=np.linspace(0, 90, 91)
        flux=np.array([flusso(ph, a) for a in alpha])
        plt.plot(alpha, flux, 'o-', color="orange")
        plt.xlabel("Angolo[°]")
        plt.ylabel("Flusso relativo")
        plt.title("Flusso relativo in funzione dell'angolo tra il Sole e lo Zenith")
        plt.show()
        


#definizione della funzione che si occupa dello studio sull'Ozono


def Stud_O3():
    """
    Studio qualitativo dell'assorbimento dell'Ozono.
    Aggiunge un quinto istogramma: angolo scelto CON ozono.
    """
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
        du=300
        T_O3 = 243
        while True:
            S=input("Inserire l'angolo per lo spessore della massa d'aria: ")
            S_fl=float(S)
            if S_fl>90 or S_fl<0:
                print("Bisogna inserire un angolo tra 0° e 90°")
            else:
                break
        ph=hm(T)
        
        
        #definizione conteggi di fotoni per solo assorbimento O3
        
        
        assO3_zen=assorb_O3(L_tot, S_z, du, T_O3)
        assO3_ang=assorb_O3(L_tot, S_theta(S_fl), du, T_O3)
        assO3_oriz=assorb_O3(L_tot, S_oriz, du, T_O3)
        
        
        #definizione conteggi di fotoni per combo Rayleigh/O3
        
        
        z=zenith(ph)*assO3_zen
        o=oriz(ph)*assO3_oriz
        f=phi(ph, S_fl)*assO3_ang
        
        
        #plot del grafico
        
        
        plt.figure(figsize=(14, 8))
        plt.bar(L_tot, ph, width=(L_tot[1]-L_tot[0]), color='indigo', alpha=0.7, label='Nessun assorbimento', linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, z, width=(L_tot[1]-L_tot[0]), color='dodgerblue', alpha=0.7, label='Zenith ', linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, f, width=(L_tot[1]-L_tot[0]), color='limegreen', alpha=0.7, label=f'Angolo scelto', linewidth=0.5, edgecolor='black')
        plt.bar(L_tot, o, width=(L_tot[1]-L_tot[0]), color='crimson', alpha=0.7, label='Orizzonte ', linewidth=0.5, edgecolor='black')
        plt.xlabel('Lunghezza d\'onda [nm]', fontsize=12)
        plt.ylabel('Conteggio ', fontsize=12)
        plt.title(f'Distribuzione osservata di fotoni con scattering Rayleigh+Assorbimento O3', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

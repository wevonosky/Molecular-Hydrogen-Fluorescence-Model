#H2 FLUORESCENCE CODE#####################################################################################
import math
from scipy.io.idl import readsav
import numpy as np
import voigt
from scipy import special
from scipy import interpolate
from scipy import integrate
import time
import warnings
from astropy.io import fits
import pickle

warnings.filterwarnings("ignore")



# These are fundamental constants in cgs
# Because they are here, they have "global" scope and can
# be "seen" by the functions and the main program.
kb = 1.380658e-16   #Boltzmann constant (erg/K)
h = 6.626075540e-27 #Planck constant (erg/s)
c = 2.99792458e+10  #Speed of light (cm/s)
me = 9.10938188e-28 #Mass of electron (g)
esu = 4.8032068e-10 #Charge of electron (statcoulombs) 
mh = 1.6733e-24     #Mass of Hydrogen atom (g)
mh2 = 2.0*mh        #Mass of Hydrogen molecule (g)

#Conversion factors
cmperang = 1.0e-8   #number of centimeters in an Angstrom
wnperev = 8065.5    #number of wavenumbers in an eV
version = 'new'     #'old' or 'new' for cos_lsf_new input

#variables defined in the Lyman Profile Generator from Allison
pc_in_cm = 3e18
au_in_cm = 1.5e13
lya_rest = 1215.67
c_km = 2.998e5

lya_rest = 1215.67
ccgs = 3e10
e=4.8032e-10            # electron charge in esu
mp=1.6726231e-24        # proton mass in grams

#Allison Youngbloods Lyman generator functions (https://github.com/allisony)#######################################################################
def tau_profile(ncols,vshifts,vdop,h1_or_d1):

    """ 
    Computes a Lyman-alpha Voigt profile for HI or DI given column density,
    velocity centroid, and b parameter.

    """

    ## defining rest wavelength, oscillator strength, and damping parameter
    if h1_or_d1 == 'h1':
        lam0s,fs,gammas=1215.67,0.4161,6.26e8
    elif h1_or_d1 == 'd1':
        lam0s,fs,gammas=1215.3394,0.4161,6.27e8
    else:
        raise ValueError("h1_or_d1 can only equal 'h1' or 'd1'!")

    Ntot=10.**ncols  # column density of H I gas
    nlam=1000.       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # function of wavelength (one side of transition)
    u_parameter=np.zeros(nlam)  # Voigt "u" parameter
    nu0s=ccgs/(lam0s*1e-8)  # wavelengths of Lyman alpha in frequency
    nuds=nu0s*vdop/c_km    # delta nus based off vdop parameter
    a_parameter = np.abs(gammas/(4.*np.pi*nuds) ) # Voigt "a" parameter -- damping parameter
    xsections_nearlinecenter = np.sqrt(np.pi)*(e**2)*fs*lam0s/(me*ccgs*vdop*1e13)  # cross-sections 
                                                                           # near Lyman line center

    wave_edge=1210. # define wavelength cut off
    wave_symmetrical=np.zeros(2*nlam-1) # huge wavelength array centered around a Lyman transition
    wave_onesided = np.zeros(nlam)  # similar to wave_symmetrical, but not centered 
                                    # around a Lyman transition 
    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter

    ## find end point for wave_symmetrical array and create wave_symmetrical array
    num_elements = 2*nlam - 1
    first_point = wave_edge
 
    mid_point = lam0s
    end_point = 2*(mid_point - first_point) + first_point
    wave_symmetrical = np.linspace(first_point,end_point,num=num_elements)
    wave_onesided = np.linspace(lam0s,wave_edge,num=nlam)

    freq_onesided = ccgs / (wave_onesided*1e-8)  ## convert "wave_onesided" array to a frequency array

    u_parameter = (freq_onesided-nu0s)/nuds  ## Voigt "u" parameter -- dimensionless frequency offset

    xsections_onesided=xsections_nearlinecenter*voigt.voigt(a_parameter,u_parameter)  ## cross-sections
                                                                                # single sided
                                                                                ## can't do symmetrical 
    
    xsections_onesided_flipped = xsections_onesided[::-1]
    ## making the cross-sections symmetrical
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided,axis=0) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all

def total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5):

    """
    Given a wavelength array and parameters (H column density, b value, and 
    velocity centroid), computes the Voigt profile of HI and DI Lyman-alpha
    and returns the combined absorption profile.

    """

    ##### ISM absorbers #####

    ## HI ##
   
    hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'h1')
    tauh1=np.interp(wave_to_fit,hwave_all,htau_all)
    clean_htau_all = np.where(np.exp(-htau_all) > 1.0)
    htau_all[clean_htau_all] = 0.0

    ## DI ##

    d1_col = np.log10( (10.**h1_col)*d2h )

    dwave_all,dtau_all=tau_profile(d1_col,h1_vel,h1_b,'d1')
    taud1=np.interp(wave_to_fit,dwave_all,dtau_all)
    clean_dtau_all = np.where(np.exp(-dtau_all) > 1.0)
    dtau_all[clean_dtau_all] = 0.0


    ## Adding the optical depths and creating the observed profile ##

    tot_tau = tauh1 + taud1
    tot_ism = np.exp(-tot_tau)

    return tot_ism

def damped_lya_profile(filename,wave,h1_col,h1_b,h1_vel,d2h):

    """
    Computes a damped (attenuated) Lyman-alpha profile (by calling the functions
    lya_intrinsic_profile_func and total_tau_profile_func) and convolves it 
    to the proper resolution.

    """
    
    #textfile=np.loadtxt(filename)
    wave_to_fit=wave
    #lya_intrinsic_profile=textfile[:,1]
    #intrinwave=textfile[:,0]
    
    textfile = fits.open(filename)
    dataattn=textfile[1].data
    
    intrinwave=dataattn['wavelength']
    lya_intrinsic_profile=dataattn['flux']
        
    pointed=np.where(lya_intrinsic_profile==max(lya_intrinsic_profile))
    wavepointed=intrinwave[pointed]
    centraled=1215.67
    shifted=np.abs(centraled-wavepointed)
    wavecorred=intrinwave-shifted
    
    function = interpolate.interp1d(wavecorred,lya_intrinsic_profile,bounds_error=False,fill_value=0.0)
    fineflux=function(wave)
    
    total_tau_profile = total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h)

    lya_obs_high = fineflux * total_tau_profile

    return lya_obs_high
    
    

#H2 Model Functions from Matt Mcjunkin####################################################################
#H2 Energy Calculator#####################################################################################
def h2ejx(v,maxj):
    E_vj = np.zeros(maxj+1)                 #Array to be populated by energies

    #Molecular energy constants
    Te = 0.0          #No electronic contribution in ground state
    we = 4401.21      #cm^-1
    wexe = -121.33    #cm^-1
    weye = 0.812      #cm^-1
    Be = 60.853       #cm^-1
    alphae = -3.062   #cm^-1
    gammae = 0.057    #cm^-1
    deltae = -0.005   #cm^-1
    De = -4.66e-2     #cm^-1
    betae = 0.18e-2   #cm^-1
    pie = -0.005e-2   #cm^-1
    He = 4.9e-5       #cm^-1
    etae = -0.1e-5    #cm^-1

    for j in range(0,maxj+1):
        Bv = Be + alphae*(v+0.5) + gammae*(v+0.5)**2 + deltae*(v+0.5)**3 #cm^-1
        Dv = De + betae*(v+0.5) + pie*(v+0.5)**2                         #cm^-1
        Hv = He + etae*(v+0.5)                                           #cm^-1

        G = we*(v+0.5) + wexe*(v+0.5)**2 + weye*(v+0.5)**3               #cm^-1
        F = Bv*j*(j+1.0) + Dv*(j**2)*(j+1.0)**2 + Hv*(j**3)*(j+1.0)**3   #cm^-1

        E_vj[j] = Te + G + F                                             #cm^-1 

#        print E_vj[j]/wnperev                                            #eV
    return E_vj

#Thermal H2 population maker (uses h2ejx)##################################################################    
def h2jpopx(t,v,maxj,maxv):
    j = np.arange(maxj + 1, dtype = float)  #Array of j values
    g_j = 2.0*j + 1.0
    E_j = h2ejx(v,maxj)                  #cm^-1

    nj = g_j*np.exp(-(E_j*h*c)/(kb*t)) #Herzberg pg. 124

    #Need to properly normalize now (follow Herzberg pgs. 123-125)
    #For most astrophysically relevant temperatures, a max j of 100
    #should be adequate for getting the Q sum to converge

    highj = 25   #TOO MUCH HIGHER AND THE NUMBERS GET SO SMALL THEY ROLLOVER
    E_i = h2ejx(v,highj)
    i = np.arange(highj + 1, dtype = float)

    Qr = (2.0*i + 1.0)*np.exp(-(E_i*h*c)/(kb*t))
    njn = nj/math.fsum(Qr)

    #Get vibrational population (only relevant for large temperatures)
    E_v = np.zeros(maxv+1)

    for m in range(0,maxv+1):
        E_v[m] = h2ejx(m,0)      #cm^-1
    Qv = np.exp(-(E_v*h*c)/(kb*t))
    nvn = Qv/math.fsum(Qv)
    njn = njn*nvn[v]
    return njn

#Python version of wavegrid#################################################################################
def wavegrid(wmin,wmax,wbin):
    wrange = wmax - wmin
    npts = long(wrange/wbin) + 1
    wavegrid = np.arange(npts, dtype = float)*wbin + wmin
    return wavegrid

#Python version of poly######################################################################################
def poly(x,c):
    n = len(c)-1
    y = c[n]
    for i in range(n-1,-1,-1):
        y = y*x + c[i]
    return y    

#My Python Voigt calculator###################################################################################
def voigt2(a,u):
    x = u/a                                                    #Define parameters for U function
    t = 1/(4.0*a**2)                                           # 
    z = (1 - complex(0,1)*x)/(2.0*np.sqrt(t))                  #
    U = np.sqrt(math.pi/(4.0*t))*special.wofz(complex(0,1)*z)  #Plug into U(x,t)
    H = U/(a*math.sqrt(math.pi))                               #Voight function
    return H

#Python version of sigma_line##################################################################################
def sigma_line(lambs,f,gamma,b_km,wave,vsh):
    lam = lambs*1.0e-8                                         #Wavelength in cm 
    lamvsh = lam*(1+(vsh*1.0e+5)/c)                            #Account for velocity shift
    nu = c/lam                                                 #Frequency
    nuvsh = c/lamvsh                                           #Frequency with velcity shift
    dnu = nu/c*b_km*1.0e+5  #Note that b-parameter is not the FWHM, nor the HWHM, nor the Gaussian sigma
    a = gamma/(4.0*math.pi*dnu)
    lams = wave*1.0e-8
    freq = c/lams
    u = np.abs(freq - nuvsh)/dnu
    h = voigt2(a,u)                                             #Calculate Voigt profile
    sig = (2.654e-2)*f*h/dnu/math.sqrt(math.pi)                #Calculate cross-section
    out = (lams*1.0e+8,sig.real)                               #Return wavelength and real part of sigma
    return out

def cos_lsf_new(lam,version):
    if lam < 1800.0:
        if lam > 1450.0:
            chan ='g160m'
        else:
            chan = 'g130m'
    else:
        chan = 'g225m'

    if (version == 'new' and ((chan == 'g130m') or (chan == 'g160m'))):
        q = readsav('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Programs/H2 Fluoresence Code/cos_lsf_new.idl')
        #print '/Users/Matt/IDLWorkspace80/COS_FIT/cos_lsf_new2.idl'
    else:
        q = readsav('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Programs/H2 Fluoresence Code/cos_lsf.idl')
        #print '/Users/Matt/IDLWorkspace80/COS_FIT/cos_lsf2.idl'

    chan = np.where(chan == q.lsfchan)
    chan = int(chan[0])

    #print np.shape(q.lsf)
    lamind = np.argmin(np.abs(q.lsfwave[chan,:]-lam))
    lsfstart = q.lsf[:,lamind,chan]
    lsf = lsfstart.T

    xarray = lam+q.lsfpix*0.001*q.lsfpixscale[chan]
    return (lsf,xarray)                   






################################################################################################################    
#THIS IS THE MAIN PROGRAM#######################################################################################
################################################################################################################
start_time = time.time()   #Time the execution of the program

#Input which star and what type of run
starname = 'GJ176'
run_type = 'Mesh_Test'

#Define the parameter space
h2column = np.arange(14,16.1,1, dtype = np.float64)
temprange = np.arange(2200, 2400, 100)
h1colden = np.arange(13.8,14.2,.1, dtype = np.float64)
#From the Find Model Lines file, input the transmission lines for botht he 1-4 and 1-7 transitions
trans14 = [1431.01,1489.57, 1504.76]
trans17 = [1467.08, 1524.65, 1556.87]

transwaves_unsorted = trans14 + trans17

#Creat meshed grid of values and index points for iteration through the model
meshh2, meshtr, meshh1 = np.meshgrid(h2column, temprange, h1colden)
meshvalues = np.vstack([meshh2, meshtr, meshh1]).reshape(3, -1).T
meshh2point, meshtrpoint, meshh1point = np.meshgrid(range(len(h2column)), range(len(temprange)), range(len(h1colden)))
meshpoints = np.vstack([meshh2point, meshtrpoint, meshh1point]).reshape(3,-1).T

#Calculate the degrees of freedom. This is equal to: # of variables - # of parameters
dof = len(transwaves_unsorted)-3
print 'The degrees of freedom are '+str(dof)
if dof == 0:
    dof = 1

#Combining and sorting the transmission lines
alltranswave = sorted(trans14+trans17)

#Calculations which provide an estimate for how long the model will run
runss = len(h2column)*len(temprange)*len(h1colden)
lengthhh = (11988/1617)*runss/3600.0

print 'The code is going to run ' + str(runss) +' times and take approx '+ str(lengthhh) + ' hours to finish'

#Loading in the Lyman Intrinsic Profile (Use whatever mechanism needed to load in the data)

#dataintrin = np.loadtxt('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Star Data/GJ832/GJ832_LyA_intrinsic_profile.txt')
#wavelengthss = dataintrin[:,0]
#flux1 = dataintrin[:,1] 

hdufits = fits.open('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Star Data/'+starname+'/'+starname+'lyman.fits')
datafits = hdufits[1].data

#Assigning variables for the wavelength and flux values from intrinsic profile
wavelengthss = datafits['wavelength']
flux1 = datafits['flux']

#Only necessary if your intrinsic profiles are shifted due to the radial velocity of the star
#We are assuming we are in a reference frame moving with the surface of the star
point = np.where(flux1 == max(flux1))
wavepoint = wavelengthss[point]
central = 1215.67 #Central wavelength of intrinsic profile
shift = np.abs(central-wavepoint)
wavecorr = wavelengthss-shift   #wavelengths shifted back from corrected values in Allisons code
wavefine = np.arange(min(wavecorr),max(wavecorr),0.005)   #Creating a fine wavelength grid

#Loading in a list of wavelengths, flux, and flux errors from obervational data
truthdata=readsav("C:/Users/Will Evonosky/Dropbox/SOARS 2016/Star Data/Nick's Line List/H2_info/"+starname+"_H2.sav")
truthflux=truthdata.flux
truthfluxerr=truthdata.flux_err
truthwaves=truthdata.centw

#Extracting only the wavelengths we want to model from the truth data
truthfluxed = []
truthfluxerred = []
truthwavesed = []
for (i,k,l) in zip(truthwaves, truthflux, truthfluxerr):
    margin = 0.20
    for j in alltranswave:
        if i <= j+margin and i >= j-margin:
            truthwavesed.append(j)
            truthfluxed.append(k)
            truthfluxerred.append(l)
if not truthwavesed == alltranswave:
    raise ValueError ("the two wavelength lists don't match")
    
truthfluxed = np.asarray(truthfluxed)
truthfluxerred = np.asarray(truthfluxerred)

#Creating array of zeros to fill with chi-square values later
chi_sqr_grid = np.zeros([len(h2column), len(temprange), len(h1colden)])
#Creating empty list to be populated later with all the model result data
alldata = []
positionalldata=runss

#Looping through the meshgrid's we made earlier
for pol in range(len(meshpoints)):
    #Variables
    Ntot=(10)**meshvalues[pol][0]    #Total column density to be thermally distributed over the v,j lines
    temp = meshvalues[pol][1]     #Gas temperature (K)
    bturb =np.sqrt((2*kb*temp)/mh2)/1.0e5       #Absorbing line-width
    vsh = 0.0         #Velocity shift of absorbtion/emission lines
    bturb_em = 17.0   #Emitting line-width
    HIcolumn= meshvalues[pol][2]      #Column density of atmoic hydrogen
    H1temp=10000 #This could easily become another varied parameter if desired
    h1bval=np.sqrt((2*kb*H1temp)/mh)/1.0e5 #b-value for H1
    
    #Compute and create the attenuated lyman alpha profile#########################################################
    
    Ly_Attn=damped_lya_profile('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Star Data/'+starname+'/'+starname+'lyman.fits',
    wavefine, HIcolumn, h1bval, 0.0,1.5e-5)
    
    #Ly_Attn=damped_lya_profile('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Star Data/GJ832/GJ832_LyA_intrinsic_profile.txt',
    #wavefine, HIcolumn, h1bval, 0.0,1.5e-5)
    #Start of Matt's Code  
        
    lam_o = 1216.0    #Central wavelength of absorbing region
    delta_lam = 0.3   #Wavelength step (Absorbing region goes from lam_o - delta_lam to lam_o + delta_lam)
    
    #v_lim = 29 #10    #Highest v value to look up to (goes up to 37 for v_up, up to 29 for v_lw)
    #j_lim = 29 #20    #Highest j value to look up to (goes up to 29 for j_up AND j_lw)
    
    v_lim=3
    j_lim=29
    
    #Read in the H2 transition parameters
    s = readsav('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Programs/H2 Fluoresence Code/fluormod_trans.idlsav')
    q = readsav('C:/Users/Will Evonosky/Dropbox/SOARS 2016/Programs/H2 Fluoresence Code/fluormod_wav.idlsav')
    
    atot = s.h2_trans.atotal
    acont = s.h2_trans.acontm
    aval = s.h2_trans.avalue
    lam = q.wavl
    elec = s.h2_trans.band
    v_up = s.h2_trans.vu
    v_lw = s.h2_trans.vl
    j_up = s.h2_trans.ju
    j_lw = s.h2_trans.jl
    lam_cm = lam*(1.0e-8)
    
    #Read in the Lyman-alpha profile of your choice, this uses the data
    lyaflux = Ly_Attn
    lyawave = wavefine
    
    
    wave = wavegrid(min(lyawave)-1,max(lyawave)+1,0.0005)                      #Creates wavelength array (start,end,interval) ***was 1210 start 1250 end***
    #cpoly = [1.5518515e-6,-2.5537233e-9,1.0506003e-12]     #Accounts for V4046 airglow emission.  Do not use
    #a = np.where((lyawave > 1215) & (lyawave < 1216.3))    #for general Lya profile!!
    #y = poly(lyawave[a],cpoly)                             
    #lyaflux[a] = where(y<3e-14,3e-14,y)                    #For flux below 3e-14, set flux value to 3e-14
    
    f = interpolate.interp1d(lyawave,lyaflux,bounds_error=False,fill_value=0.0)
    flux = f(wave)
    
    upflux = np.zeros([v_lim+1,j_lim+1],dtype=float)   #Define arrays to be populated with absflux = wave*0.0
    absflux = wave*0.0                              #emission and absorption fluxes
    
    em_wave = wavegrid(900,1800,0.00997) #Wave grid for emission line flux calculation
    em_flux = em_wave*0.0                                               #Flux grid for emission line flux calculation 
    
    g_up = ( (2.*j_up) + 1. )                                           #Upper state level degeneracy
    g_lw = ( (2.*j_lw) + 1. )                                           #Lower state level degeneracy 
    
    #Creation of oscillator strengths (both abs and emis)
    phys_const = (me*c)/(8.0*(math.pi*esu)**2)
    f_abs = phys_const * (g_up/g_lw) * (lam_cm**2) * aval    #Absorption oscillator strength
    f_emis = (g_lw/g_up) * f_abs                            #Emission oscillator strength  
    
    #Select absorbing transistions in the region of the Lya profile
    #hunt_ly = np.where( (lam >= (lam_o - delta_lam)) & (lam <= (lam_o + delta_lam)) & \ 
    #          (elec == 'Ly') & \                                                       
    #          (v_up <= v_lim) & (v_lw <= v_lim) & \                                    
    #          (j_lw <= j_lim) & (j_up <= j_lim) )
    
    hunt_ly = np.where( (lam >= (lam_o - delta_lam)) & (lam <= (lam_o + delta_lam))  #Lambda condition
                        #& ((elec == 'Ly') or (elec eq 'Wp')) #or (elec eq 'Wm'))      #Band condition
                        & ((v_up <= v_lim) & (v_lw <= v_lim))                        #Vibrational condition
                        & ((j_lw <= j_lim) & (j_up <= j_lim)) )                      #Rotational condition
        
    absind = np.arange(len(lam[hunt_ly]), dtype = float)     #Array of absorption line indices
    tau = np.zeros([len(wave),len(absind)],dtype=float)      #Initialized array for optical depths      
    tautot = np.zeros(len(wave),dtype=float)                 #Initialized array for total optical depth
    tau_corr = np.zeros([len(wave),len(absind)],dtype=float) #Initialized array for corrected optical depths
    
    for i in range(0,len(absind)):
        v_up_ly = v_up[hunt_ly]
        j_up_ly = j_up[hunt_ly]
        v_lw_ly = v_lw[hunt_ly]
        j_lw_ly = j_lw[hunt_ly]
        f_abs_ly = f_abs[hunt_ly]
        lam_ly = lam[hunt_ly]
        elec_ly = elec[hunt_ly]
                        
        #The index absind[i] chooses a particular absorbing transition.  Hunt then picks all the
        #resulting emitting transitions and their constants.
        hunt = np.where( (v_up == v_up_ly[absind[i]]) & (j_up == j_up_ly[absind[i]]) & (elec == 'Ly') & (v_lw <= v_lim) )
    
        diff = lam[hunt] - lam_o
        h_atot = atot[hunt]
        h_acont = acont[hunt]
        h_aval = aval[hunt]
        h_lam = lam[hunt] 
        h_elec = elec[hunt]
        h_v_up = v_up[hunt]
        h_v_lw = v_lw[hunt]
        h_j_up = j_up[hunt]
        h_j_lw = j_lw[hunt]
        h_f_emis = f_emis[hunt] 
        h_f_abs = f_abs[hunt]
        h_cm = lam_cm[hunt]
    
        #Pick out absorbing line properties
        lam_abs = lam_ly[absind[i]]
        ftemp_abs = f_abs_ly[absind[i]]
        j_abs = j_lw_ly[absind[i]]
        v_abs = v_lw_ly[absind[i]]
        j_em = j_up_ly[absind[i]]
        v_em = v_up_ly[absind[i]]
        band = elec_ly[absind[i]]
        #print lam_abs,ftemp_abs,band,v_abs,j_abs,v_em,j_em
    
        #Calculate absorption line
        sig = sigma_line(lam_abs,ftemp_abs,1.0e8,bturb,wave,vsh)
        njn = h2jpopx(temp,v_abs,j_lim,v_lim)
        tau[:,i] = Ntot*njn[j_abs]*sig[1]
        tautot = tautot + tau[:,i]
    
    for i in range(0,len(absind)):
        tau_corr[:,i] = tau[:,i]*(tau[:,i]/tautot)

    absinfo=[]
    absinfo.append(wave.tolist())
    absinfo.append(flux.tolist())
    totfluxex=[]
    for i in range(0,len(absind)):
        v_up_ly = v_up[hunt_ly]
        j_up_ly = j_up[hunt_ly]
        v_lw_ly = v_lw[hunt_ly]
        j_lw_ly = j_lw[hunt_ly]
        f_abs_ly = f_abs[hunt_ly]
        lam_ly = lam[hunt_ly]
        
        #Pick out absorbing line properties again
        lam_abs = lam_ly[absind[i]]
        ftemp_abs = f_abs_ly[absind[i]]
        j_abs = j_lw_ly[absind[i]]
        v_abs = v_lw_ly[absind[i]]
        j_em = j_up_ly[absind[i]]
        v_em = v_up_ly[absind[i]]
        
        trans = np.exp(-tau_corr[:,i])
        absflux = (1.0-trans)*flux
        intflux = integrate.simps((1-trans)*flux,wave)
        totfluxex.append(intflux)
        upflux[v_em,j_em] += intflux
        absinfo.append(absflux.tolist())
        absinfo.append(trans.tolist())
            
    #For each upper (v,j) level, compute flux into each line out of that level
    countup = 0
    branchflux = []
    branchwaves = []
    for v in range(0,v_lim+1):
        for j in range(0,j_lim+1):
            if upflux[v,j] != 0.0:
                #(V,J) CHOOSES A PARTICULAR EXCITED STATE.  HUNT THEN PICKS OUT ALL THE RESULTING
                #EMITTING TRANSITIONS AND THEIR CONSTANTS
                v_lim=14
                hunt = np.where((v_up == v) & (j_up == j) & (elec == 'Ly') & (v_lw <= v_lim))  #<= or < for v_lw?
                diff = lam[hunt] - lam_o
                h_atot = atot[hunt]
                h_aval = aval[hunt]
                h_lam = lam[hunt]
                h_f_emis = f_emis[hunt]
                vj_branch = h_aval/h_atot
                branchflux.append(vj_branch*totfluxex[countup])
                branchwaves.append(h_lam)
                countup += 1
                v_em = v
                j_em = j
                vj_branch_tot = 0.0
                for m in range(0,len(h_lam)):
                    sig = sigma_line(h_lam[m],h_f_emis[m],1.0e+8,bturb_em,em_wave,vsh)
                    siga = sig[1]/integrate.simps(sig[1],sig[0])
                    emline = siga*upflux[v,j]*vj_branch[m]
                    em_flux += emline
                    vj_branch_tot += vj_branch[m]
                
    ##### Pulling out the flux values for the lines in order determiend by lists defined by trans14 and trans 17 ######
    branchwaves14 = branchwaves[0]
    branchwaves17 = branchwaves[1]
    branchflux14 = branchflux[0]
    branchflux17 = branchflux[1]

    waves14 = []
    waves17 = []
    
    for (i,j) in zip(branchwaves14, branchwaves17):
        waves14.append(round(i, 2))
        waves17.append(round(j,2))
        
    flux14_need = []
    flux17_need = []
    
    for i in trans14:
        flux14_need.append(branchflux14[waves14.index(i)])
        
    for j in trans17:
        flux17_need.append(branchflux17[waves17.index(j)])
    
    
    flux_from_all = np.asarray(flux14_need+flux17_need)
    
    flux_all_sorted = [x for (y,x) in sorted(zip(transwaves_unsorted,flux_from_all))]

    #Calculating the chi-square value and inserting it into array
    chisq = np.sum(((truthfluxed - flux_all_sorted) / truthfluxerred)**2)
    print 'The reduced chi value is '+str(chisq/dof)
    chi_sqr_grid[meshpoints[pol][0], meshpoints[pol][1], meshpoints[pol][2]]=chisq
    
   #Saving all the relevant data to one big list 
    alldata.append([meshvalues[pol][0], meshvalues[pol][1], meshvalues[pol][2], trans14, flux14_need, trans17, flux17_need])
    positionalldata -= 1
    
    print "There are " +str(positionalldata)+ ' runs left'
        

#Saving both the data structures for use later   
with open('Final_Data_'+starname+'_'+run_type+'_Run', 'wb') as f: 
    pickle.dump(alldata, f) 
    
np.save(starname+'_Chisqr_Info_'+run_type+'_Run', chi_sqr_grid)                
print 'Code ran for ',round((time.time() - start_time)/3600,2), " hours for "+ str(runss)+ ' runs'






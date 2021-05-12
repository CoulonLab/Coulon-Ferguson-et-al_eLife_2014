#############################################
#
#    Fluctuation analysis code use in [Coulon, Ferguson et al. 2014, eLife,
#    DOI: 10.7554/eLife.03939]
#
#    Copyright (C) 2014,2021  Antoine Coulon.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#    If using this code in a publication, please cite
#    [DOI: 10.7554/eLife.03939] and/or [DOI: 10.1016/bs.mie.2016.03.017]
#
#    Contact: software@coulonlab.org
#
#############################################

import os, sys, re, imp
from scipy import *
from misc import objFromDict

def setDataPath(raw,proc):
  globals()['rawDataPath']=os.path.expanduser(raw)+'/'*(raw[-1]!='/')
  globals()['procDataPath']=os.path.expanduser(proc)+'/'*(proc[-1]!='/')


##################################
##################################
version='0.6.2'
#setDataPath('~/res/Splicing/AcquisitionData/','~/res/Splicing/data/')
##################################
##################################


if not 'verbose' in globals(): verbose=1;


##################################
### Numerical computations

### Compute all crosscorrelations G
def compG_multiTau(v,t,n=8,ctr=1):
  """v: data vector (channels=rows), t: time, n: bin every n steps.\n--> Matrix of G, time vector"""
  def compInd(v1,v2):
    tau=[]; G=[]; t0=t*1.; i=0; dt=t0[1]-t0[0]
    while i<t0.shape[0]:
      tau.append(i*dt); G.append(mean(v1[:v1.shape[0]-i]*v2[i:]));
      if i==n:
        i=i//2; dt*=2; t0,v1,v2=c_[t0,v1,v2][:(t0.shape[0]//2)*2].T.reshape(3,-1,2).mean(2)
      i+=1
    return array([tau,G]).T
  if ctr: vCtr=((v.T-mean(v,1)).T);
  else: vCtr=v
  res=array([[ compInd(v1,v2) for v2 in vCtr] for v1 in vCtr])
  return ( res[:,:,:,1].T /(dot(mean(v,1).reshape(-1,1),mean(v,1).reshape(1,-1)))).T, res[0,0,:,0]


### Average and boostrap replicates
def avgBootstrap(data,nBs=3000,q12=0,normG0=0,offsetTail=0,tau=None,saveFcs4=1):
  """nBs: number of repeat in bootstrap procedure.
  q12: quantiles of confidence intervals, e.g. q12=(.05,.95). Return standard error if q12=0.
  normG0: defines if correlation functions are normalized by G_rg(0).
  offsetTail: delay window used to offset the correlation functions to 0. No offset if offsetTail=0.
  tau: Vector of tau values to use. If tau=None, uses the longest of data[i].tau."""
  def meanDiffLength(data):
    v=array([[[interp(tau,d.tau,d.G[i,j],right=0.) for j in [0,1]] for i in [0,1]] for d in data])
    sumR=zeros(v.shape[1:-1]+(tau.shape[0],)); nb=zeros(v.shape[1:-1]+(tau.shape[0],))
    for i in range(len(data)):
      d=data[i]; weight=maximum((hasattr(d,'frameWindow') and (d.frameWindow[1]-d.frameWindow[0]) or d.t.shape[0])*d.dt-tau,0)
      #print(weight[:11]/5.)
      sumR+=v[i]*weight
      nb+=weight
      #sumR[:,:,:v.shape[-1]-1]+=v[i,:,:,:-1]
      #nb[:,:,where(v[i,0,0]!=0.)[0]]+=1
    #print(len(data), nb[0,0,:11]/5.)
    return sumR/(nb+1*(nb==0))
    #return (sumR/(nb+1*(nb==0)))*(nb!=0)-100*(nb==0)

  def fbs(a):
    tmpG=meanDiffLength(a)
    if offsetTail!=0:
      tmp=tmpG[:,:,where((offsetTail[0]<=tau)*(tau<=offsetTail[1]))[0]]
      o_rr=mean(tmp[0,0,:]); o_gg=mean(tmp[1,1,:]); o_rggr=mean(r_[tmp[0,1,:],tmp[1,0,:]]);
      tmpG=(tmpG.T-array([[o_rr,o_rggr],[o_rggr,o_gg]])).T
    if normG0!=0:
      #tmp1G0=1./((tmpG[0,1,0]+tmpG[1,0,0])/2.) # Normalize by Grg(0)
      tmp1G0=1./((tmpG[0,1,1]+tmpG[1,0,1])/2.) # Normalize by (Grg(-dt)+Grg(dt))/2
      #tmp1G0=1./tmpG[0,1].max() # Normalize by max of Grg
      tmpG*=tmp1G0
    return tmpG

  if verbose: (sys.stdout.write('** Averaging and bootstrapping... '),sys.stdout.flush());

  if tau==None:
    tmp=min([d.tau[1] for d in data]); tau=data[argmax([d.tau.shape[0]*(d.tau[1]==tmp) for d in data])].tau
  if not prod([prod(d.tau==tau[:d.tau.shape[0]]) for d in data]): sys.stdout.write("(Tau vectors are not all identical) "); sys.stdout.flush(); 
  tau2=(tau[1:]+tau[:-1])/2;

  allBs=[fbs([data[j] for j in random.randint(0,len(data),len(data))]) for i in range(nBs)]
  dresG  =sort(array([a for a in allBs]),0);
  if type(q12)==tuple:
    bsG=array([fbs(data), dresG[floor(q12[0]*( dresG.shape[0]-1))], dresG[ceil(q12[1]*(dresG.shape[0]-1))]])
    bsGq=array([bsG[1]-bsG[0], bsG[2]-bsG[0]])
  else:
    bsGsd=[fbs(data),var(dresG,0)**.5];
    bsG=array([bsGsd[0], bsGsd[0]-bsGsd[1], bsGsd[0]+bsGsd[1]])

  # Save '_avg.fcs4' file
  if 'poolName' in globals() and saveFcs4:
    fn=re.escape(procDataPath+poolName+'_avg.fcs4');
    if type(q12)==tuple:
      savetxt('tmp.txt',c_[tau,bsG[0][0,0],bsG[0][1,1],bsG[0][0,1],bsG[0][1,0], bsGq[0][0,0], bsGq[0][1,1], bsGq[0][0,1], bsGq[0][1,0], bsGq[1][0,0], bsGq[1][1,1], bsGq[1][0,1], bsGq[1][1,0]],'%12.5e  ');
      os.system('echo "#Tau (in s)     Grr            Ggg            Grg            Ggr            1st q. Grr     1st q. Ggg     1st q. Grg     1st q. Ggr     2nd q. Grr     2nd q. Ggg     2nd q. Grg     2nd q. Ggr" > '+fn+'; cat tmp.txt >> '+fn)
    else:
      savetxt('tmp.txt',c_[tau,bsG[0][0,0],bsG[0][1,1],bsG[0][0,1],bsG[0][1,0], bsGsd[1][0,0], bsGsd[1][1,1], bsGsd[1][0,1], bsGsd[1][1,0]],'%12.5e  ');
      os.system('echo "#Tau (in s)     Grr            Ggg            Grg            Ggr            sigma Grr      sigma Ggg      sigma Grg      sigma Ggr" > '+fn+'; cat tmp.txt >> '+fn)

  if verbose: (sys.stdout.write('Done.\n'),sys.stdout.flush());
  res={'tau':tau,'bsG':bsG,'q12':q12,'normG0':normG0,'offsetTail':offsetTail}
  if 'poolName' in globals(): res['poolName']=poolName
  return objFromDict(**res)




#################################

def readMetadata(pathToFile,nbKeys=1,nbLines=500):
  import random as random2, string
  tempFName='tmp-'+''.join(random2.choice(string.ascii_letters+string.digits) for x in range(10))+'.txt'
  true=True; false=False; null=None;
  aa=open(pathToFile,'r')
  tmp=[aa.readline().replace('true','True').replace('false','False') for ii in range(nbLines)];
  open(tempFName,'w').writelines(['# -*- coding: utf-8 -*-\nglobal tmpMD; tmpMD={\r\n']+tmp[1:where(array([b[0]=='}' for b in tmp]))[0][nbKeys-1]]+['}}\r\n']);
  tmpMD=imp.load_source('',tempFName).tmpMD;
  os.system('rm '+tempFName)
  if nbKeys==1: return objFromDict(**tmpMD['Summary'])
  else: return objFromDict(**tmpMD)


# Load experimental data
ppath=(lambda path,bb: (bb[0] not in ['/','~'])*path+os.path.expanduser(bb))

def loadExpData(fn,nMultiTau=8):
  global poolName
  if fn[-3:]=='.py': fn=fn[:-3]
  if fn[-5:]!='.list': fn=fn+'.list'
  poolName=fn[:-5]
  fn=procDataPath+fn
  fn=os.path.expanduser(fn)
  if fn in sys.modules: reload(sys.modules[fn])
  inFiles=imp.load_source('',fn+'.py')
  lf=[objFromDict(**a) for a in inFiles.listFiles]
  data=[]
  if verbose: (sys.stdout.write('** Loading experimental data:\n'),sys.stdout.flush());
  for i in range(len(lf)):
    a=lf[i]
    if verbose: (sys.stdout.write('     file %d/%d "%s..."\n'%(i+1,len(lf),a.trk_r.replace('_green.trk',''))),sys.stdout.flush());
    data.append(objFromDict(**{}))
    data[-1].path=procDataPath
    if hasattr(a,'trk_r'):  data[-1].trk_r=loadtxt(ppath(procDataPath,a.trk_r))
    if hasattr(a,'trk_g'):  data[-1].trk_g=loadtxt(ppath(procDataPath,a.trk_g))

    if hasattr(a,'detr'): # Columns of detr: frame, red mean, red sd, green mean, green sd, red correction raw, red correction polyfit, green correction raw, green correction polyfit
      data[-1].detr=loadtxt(ppath(procDataPath,a.detr))
      rn=data[-1].detr[:,2]/data[-1].detr[0,2]; x=where(abs(diff(rn))<.1)[0]; pf=polyfit(x,log(rn[x]),8)
      rf=exp(sum([data[-1].detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
      gn=data[-1].detr[:,4]/data[-1].detr[0,4]; x=where(abs(diff(gn))<.1)[0]; pf=polyfit(x,log(gn[x]),8)
      gf=exp(sum([data[-1].detr[:,0]**ii*pf[-1-ii] for ii in range(len(pf))],0))
      data[-1].detr=c_[data[-1].detr,rn,rf,gn,gf]

    if hasattr(a,'frameWindow'):  data[-1].frameWindow=a.frameWindow
    if hasattr(a,'actualDt'): data[-1].actualDt=data[-1].dt=a.actualDt
    else: data[-1].dt=0
    if hasattr(a,'hrsTreat'): data[-1].hrsTreat=a.hrsTreat
    if hasattr(a,'hrsInduc'): data[-1].hrsInduc=a.hrsInduc

    if hasattr(a,'rawPath'): data[-1].rawPath=ppath(rawDataPath,a.rawPath)
    if hasattr(a,'rawTrans'): data[-1].rawTrans=a.rawTrans

    if hasattr(a,'fcs_rr'): data[-1].fcs_rr=loadtxt(ppath(procDataPath,a.fcs_rr),skiprows=7)
    if hasattr(a,'fcs_gg'): data[-1].fcs_gg=loadtxt(ppath(procDataPath,a.fcs_gg),skiprows=7)
    if hasattr(a,'fcs_rg'): data[-1].fcs_rg=loadtxt(ppath(procDataPath,a.fcs_rg),skiprows=7)
    if hasattr(a,'fcs_gr'): data[-1].fcs_gr=loadtxt(ppath(procDataPath,a.fcs_gr),skiprows=7)

    if hasattr(a,'trk_r'):  data[-1].name=a.trk_r.replace('_red.trk','').replace('.trk','')

    if hasattr(a,'ctrlOffset'):  data[-1].ctrlOffset=array(a.ctrlOffset)
    if hasattr(a,'transfLev'):  data[-1].transfLev=array(a.transfLev)

    if hasattr(a,'maxProj'):
      data[-1].maxProj=ppath(procDataPath,a.maxProj)
      if not os.path.exists(data[-1].maxProj): print("!! Warning: file '%s' does not exist."%(a.maxProj))

    if hasattr(a,'metadata'):
      data[-1].metadata=readMetadata(ppath(procDataPath,a.metadata))
      if data[-1].dt==0: data[-1].dt=data[-1].metadata.Interval_ms/1000.
      #else: print("Using provided dt=%fs, not %fs from metadata."%(data[-1].dt,data[-1].metadata.Interval_ms/1000.))
    else:
      print("!! Warning: No metadata and no dt provided. Using dt=1."); data[-1].dt=1.

    data[i].t=data[i].trk_r[:,3]*data[-1].dt;
    data[i].r=data[i].trk_r[:,2];
    data[i].g=data[i].trk_g[:,2];
    #if hasattr(data[i],'detr'): # Detrending from s.d.
    #  data[i].r=data[i].r/data[i].detr[:,5]
    #  data[i].g=data[i].g/data[i].detr[:,7]
    if hasattr(data[i],'detr'): # Detrending from s.d. polyfit
      data[i].r=data[i].r/data[i].detr[:,6]
      data[i].g=data[i].g/data[i].detr[:,8]

    if not hasattr(data[-1],'frameWindow'): data[-1].frameWindow=[0,data[-1].t.shape[0]]


  if verbose: (sys.stdout.write('     Done.\n'),sys.stdout.flush());

  # Recompute correlations
  if nMultiTau!=0:
    if verbose: (sys.stdout.write('** Recomputing correlations functions... '),sys.stdout.flush());
    for i in range(len(data)):
      d=data[i]
      d.fcsRecomp=True
      d.G,d.tau=compG_multiTau(c_[d.r,d.g][d.frameWindow[0]:d.frameWindow[1]].T,d.t[d.frameWindow[0]:d.frameWindow[1]],nMultiTau)
      # Write .fcs4 files
      fn=re.escape(d.path+d.name+'.fcs4');
      savetxt('tmp.txt',c_[d.tau,d.G[0,0],d.G[1,1],d.G[0,1],d.G[1,0]],'%12.5e  ');
      os.system('echo "#Tau (in s)     Grr            Ggg            Grg            Ggr" > '+fn+'; cat tmp.txt >> '+fn)
    if verbose: (sys.stdout.write('Done.\n'),sys.stdout.flush());
  else:
    for i in range(len(data)):
      data[i].fcsRecomp=False
      data[i].tau=data[i].fcs_rr[:,0]*data[i].dt/data[i].fcs_rr[0,0];
      #data[i].tau=data[i].fcs_rr[:,0]
      data[i].G=array([[d.fcs_rr[:,1],d.fcs_rg[:,1]],[d.fcs_gr[:,1],d.fcs_gg[:,1]]])
  
  return data





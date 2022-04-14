from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
import kivy
from kivy.app import App
from kivy.uix.button import Button
import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle
#from kivy.graphics import Color
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

from kivy.properties import ListProperty, ObjectProperty
from kivy.graphics.vertex_instructions import (Rectangle, Ellipse, Line)
from kivy.graphics.context_instructions import Color
from kivy.core.window import Window



def ActiveSetPrimal(x,R):
    #===================================================#
    #Implements the Active Primal Set algorithm rom Nocedal and Wright
    #Solved inequality constrained QP, by sequential equality constrained QP
    #We are solving an example problem rom Modern Portolio theory
    
    #Takes x an initial guess portolio o stocks, and return R
    
    #===================================================#
    
    
    mu1 = 15.1
    mu2 = 12.5
    mu3 = 14.7
    mu4 = 9.02
    mu5 = 17.68

    mu = np.array([[mu1],[mu2],[mu3],[mu4],[mu5]])
    AE = np.array([[1,1,1,1,1],
                [mu1,mu2,mu3,mu4,mu5]])
    AI = np.eye(5)

    A = np.concatenate((AE,AI),axis=0)
    #print("A")
    #print(A)

    G = np.array([[2.30,0.93,0.62,0.74,-0.23],
                   [0.93,1.4,0.22,0.56,0.26],
                   [0.62,0.22,1.8,0.78,-0.27],
                   [0.74,0.56,0.78,3.4,-0.56],
                   [-0.23,0.26,-0.27,-0.56,2.6]])

    c = np.array([[0],[0],[0],[0],[0]])
    #x = np.array([[0.2],[0.2],[0.2],[0.2],[0.2]])
    #x = np.array([[0],[0.1],[0.5],[0.1],[0.3]])
    #R = np.dot(mu.T,x)
    bE = np.array([[1],[R]])
    bI = np.array([[0],[0],[0],[0],[0]])

    b = np.concatenate((bE,bI),axis=0)


        
    #Projected CG method:
    Wk = [0,1]
    #Wk = Wk.append(1)
    #Wk = Wk.append(2)
    I = [2,3,4,5,6]

    #print("Len Wk", len(Wk))
    alphak = 0.1
    #print("START ITERATING")
    for k in range(100):
        #find pk 16.39
        Wsize = len(Wk)
        AW = np.zeros((Wsize,5))
        for j in range(Wsize):
            AW[j,:] = A[Wk[j],:]
        
        K = np.zeros((5+Wsize,5+Wsize))
        
        K[0:5,0:5] = G
        K[5:5+Wsize,0:5] = AW
        K[0:5,5:5+Wsize] = AW.T
        
        Kinv = np.linalg.inv(K)
        

        b1639 = np.zeros((Wsize,1))
        for i in range(Wsize):
            b1639[i] = b[Wk[i]]
        h = np.dot(AW,x)-b1639
        g = c+np.dot(G,x)

       # print(h)
       # print(g)

        RHV = np.concatenate((g,h),axis=0)
        #print(RHV)

        LHV = np.dot(Kinv,RHV)
        #print(LHV)
        p = -LHV[0:5]

        
        #print(xstar)
        #p size 5
        Boolcheck = np.abs(p) <= 1e-10
        if np.all(Boolcheck == True):
            #print("In if pk == 0")
            lambdas = LHV[5:]
            
            #print("lambdas")
            #print(lambdas)
            Boolcheck = lambdas >= 0
            #print(Boolcheck)
            Ichecks = []
            for i in Wk:
                if i in I:
                    Ichecks.append(i)
                    #Ichecks now have indices between 0 to 6...
            #print(len(lambdas),Ichecks)        
            #print("Ichecks ", Ichecks)
            if len(Ichecks) > 0:
                alllambdas = True
                for i in Ichecks:
                    indval = np.where(Wk == i)[0]
                    if Boolcheck[indval] == False:
                       alllambdas = False
                 
                if alllambdas == True:
                    #STOP ITERATION, WE FOUND SOLUTION
                    #print("ALL LAMBDAS lambdai >=0 FOUND SOLUTION BREAK")
                    break
            else:
                
                
                #Find the lowest lambda of Wk and I, and return the index...
                if len(Ichecks) > 0:
                    lambdachecks = []
                    for i in Ichecks:
                        lambdachecks.append(lambdas[i])
                    jpre = np.argmin(lambdachecks)
                    j = Icheck[jpre]
                    
                    x = x #no update to x, basically
                    Wk.remove(j) #remove j from Wk
        
        else:
            #pk is not 0
            #print("In else")
            #compute ak from 16.41
          
            
            #Getting constraints that are not in working set
            notWk = []
            for i in range(7):
                if i not in Wk:
                    notWk.append(i)
            
            notWkandneg = []
            for i in notWk:
                aiTpk = np.dot(A[i,:],p)
                if aiTpk < 0:
                   notWkandneg.append(i)
            
               
            #Hvis ak = 1, og der ikke er nogen nye active constraints, så er der ingen blocking constraints
            
            
            #Make list of RHS
            #then choose ak = smallest of RHS, or 1
            RHSlist = np.zeros(len(notWkandneg))
            kcount = 0
            for i in notWkandneg:
                RHS = (b[i]-np.dot(A[i,:],x))/np.dot(A[i,:],p)
                RHSlist[kcount] = RHS
                kcount+=1
            akRHSmin = min(RHSlist)
            
            if akRHSmin < 1:
                ak = akRHSmin
            else:
                ak = 1
            
            alphak = ak
            x = x+alphak*p
            
            
            #Add working constraints blocking constraints...
            #blocking constraints skal tage udgangspunkt i den NYE x = x+alphak*p, så det skal være HER, EFTER x = x+alphak*p update
            
            #Check constraints not in Wk currently
            for i in notWk:
                ci = np.dot(A[i,:],x)-b[i]
                if ci == 0:
                    #add new blocking constraint i to Wk, i think
                    Wk.append(i)
                    break
            
    return x

def Var(x):
        #Returns the variance of the found solution asset vector x
		G = np.array([[2.30,0.93,0.62,0.74,-0.23],
			[0.93,1.4,0.22,0.56,0.26],
			[0.62,0.22,1.8,0.78,-0.27],
			[0.74,0.56,0.78,3.4,-0.56],
			[-0.23,0.26,-0.27,-0.56,2.6]])
		       
		xTGx = np.dot(x.T,np.dot(G,x))
		return xTGx    


def EfficientFrontier():
	mu1 = 15.1
	mu2 = 12.5
	mu3 = 14.7
	mu4 = 9.02
	mu5 = 17.68

	mu = np.array([[mu1],[mu2],[mu3],[mu4],[mu5]])
	#x = np.array([[0],[0.1],[0.5],[0.1],[0.3]])
	#x = np.array([[0],[0.1],[0.5],[0.1],[0.3]])
	x = np.array([[0.2],[0.2],[0.2],[0.2],[0.2]])
	R = np.dot(mu.T,x)    
	x = ActiveSetPrimal(x,R)
	#print(x)
	

	
	Rspace = np.linspace(9.02,17.68,100)
	Varvals = np.zeros(100)
	for i in range(100):
		R = Rspace[i]
	    #linear_constraint = LinearConstraint([[1, 1,1,1,1],[15.1,12.5,14.7,9.02,17.68]], [1,R], [1,R])
	    #res = minimize(Var, x0, method='trust-constr',constraints=[linear_constraint],options={'verbose': 1}, bounds=bounds)
	    #print(res.x)
		x = np.array([[0.2],[0.2],[0.2],[0.2],[0.2]])
		x = ActiveSetPrimal(x,R)
		Varvals[i] = Var(x)
	
	return [Varvals,Rspace]



class LabelWidget(Label):
	def __init__(self,**kwargs):
		super(LabelWidget,self).__init__(**kwargs)
		
		#self.canvas.Label(text="test2",pos=(300,300),font_size=12,color=(1,1,1,1))
		#with self.canvas:
		#	TextInput(text="test",pos=(300,300),font_size=12,color=(1,1,1,1))
		#self.add_widget(Label(text="test1"))
		WindowSize = Window.size
		Color=(1,1,1,1)
			
		xofst = WindowSize[0]/5
		yofst = WindowSize[1]/4
			
		xdatascale = (2/3)*WindowSize[0]
		ydatascale = (2/3)*WindowSize[1]
		
		L1 = Label(text="MPT - Efficient Frontier",pos=(xofst+(1/3)*xdatascale,yofst+ydatascale))
		L2 = Label(text="Risk x^TGx",
			height=40,
			pos=(xofst+(1/2)*xdatascale-20,yofst-50))
			
		L3 = Label(text="Return",
			height=40,
			pos=(xofst-110,yofst+(1/2)*ydatascale+10))
			
		L4 = Label(text="x^T\mu",
			height=40,
			pos=(xofst-110,yofst+(1/2)*ydatascale-20))	


		self.add_widget(L1)
		self.add_widget(L2)
		self.add_widget(L3)
		self.add_widget(L4)
		
class DataWidget(RelativeLayout):
	def __init__(self,**kwargs):
		super(DataWidget,self).__init__(**kwargs)
		
		with self.canvas:
		
			WindowSize = Window.size
			Color=(1,1,1,1)
			
			
			#layout = GridLayout(cols=2)
			#layout.add_widget(TextInput(text="test1"))
			#layout.add_widget(TextInput(text="test2"))
			#L1 = Label(font_size=12)       
			#L1.text = 'This is some nice random text\nwith linebreak'
			#L1.texture_update()
			#L1.pos=(300,300)
			
			xofst = WindowSize[0]/5
			yofst = WindowSize[1]/4
				
			xdatascale = (2/3)*WindowSize[0]
			ydatascale = (2/3)*WindowSize[1]
			
			mu1 = 15.1
			mu2 = 12.5
			mu3 = 14.7
			mu4 = 9.02
			mu5 = 17.68
			
			mu = np.array([[mu1],[mu2],[mu3],[mu4],[mu5]])
			Results = EfficientFrontier()
			#x = xofst+100*Results[0]
			#y = -yofst+30*Results[1]
			x = Results[0]
			y = Results[1]
			xmean = np.mean(x)
			xmax = np.max(x)
			xmin = np.min(x)
			ymean = np.mean(y)
			ymin = np.min(y)
			ymax = np.max(y)
			x = (x -xmin)/(xmax-xmin)*xdatascale + xofst
			y = (y -ymin)/(ymax-ymin)*ydatascale + yofst
			#x = np.array([0,20,110,190,270,400])
			#y = np.array([0,49,100,200,259,390])
			
			#ListIt = [x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]]
			ListIt = []
			for i in range(len(x)):
				ListIt.append(x[i])
				ListIt.append(y[i])
			Color=(1,1,1,1)
			Line(points=ListIt,width=2)
			Color=(1,1,1,1)
			#Line(points=[xofst,yofst,xofst+400,yofst],width=2)
			#Line(points=[xofst,yofst,xofst,yofst+400],width=2)
			Line(points=[xofst,yofst,xofst+xdatascale,yofst],width=2)
			Line(points=[xofst,yofst,xofst,yofst+ydatascale],width=2)
			for i in range(5):
				xf = np.zeros(5)
				xf[i] = 1
				Vari = Var(xf)
				#Vari = 
				#plt.scatter(Vari,mu[i],color="black")
				Ellipse(pos=(xdatascale*(Vari-xmin)/(xmax-xmin)+xofst,yofst+ydatascale*(mu[i]-ymin)/(ymax-ymin)),size=(10,10))
		
			Color=(1,1,1,1)

				
			TextInput(text=str(round(ymin)),
			height=40,
			pos=(xofst-50,yofst),
			background_color=(0,0,0,0),
			foreground_color=(1,1,1,1))

			TextInput(text=str(round(ymax)),
			height=40,
			pos=(xofst-50,yofst+ydatascale),
			background_color=(0,0,0,0),
			foreground_color=(1,1,1,1))

			TextInput(text=str(round(xmin)),
			height=40,pos=(xofst,yofst-40),
			background_color=(0,0,0,0),
			foreground_color=(1,1,1,1))

			TextInput(text=str(round(xmax)),
			height=40,
			pos=(xofst+xdatascale-40,yofst-40),
			background_color=(0,0,0,0),
			foreground_color=(1,1,1,1))
			
			#Make cartesian grid lines
			Line(points=[xofst-5,yofst,xofst+5,yofst],width=2)
			Line(points=[xofst-5,yofst+(1/3)*ydatascale,xofst+5,yofst+(1/3)*ydatascale],width=2)
			Line(points=[xofst-5,yofst+(2/3)*ydatascale,xofst+5,yofst+(2/3)*ydatascale],width=2)
			Line(points=[xofst-5,yofst+(3/3)*ydatascale,xofst+5,yofst+(3/3)*ydatascale],width=2)
			
			Line(points=[xofst+(1/3)*xdatascale,yofst+5,xofst+(1/3)*xdatascale,yofst-5],width=2)
			Line(points=[xofst+(2/3)*xdatascale,yofst+5,xofst+(2/3)*xdatascale,yofst-5],width=2)
			Line(points=[xofst+(3/3)*xdatascale,yofst+5,xofst+(3/3)*xdatascale,yofst-5],width=2)
			


class MyGrid(GridLayout):
	def __init__(self,**kwargs):
		super(MyGrid,self).__init__(**kwargs) 		
		self.cols=3
		self.rows=6
		self.row_force_default=True
		self.row_default_height=40
		#self.add_widget(Label(text="Test1"))
		#self.add_widget(Label(text="Test2"))
		self.pos_hint={'x':0.15,'y':-0.80}
		self.size_hint=(0.6, 1)
		WindowSize = Window.size

		xofst = WindowSize[0]/5
		yofst = WindowSize[1]/4
			
		xdatascale = (2/3)*WindowSize[0]
		ydatascale = (2/3)*WindowSize[1]
		
		G = np.array([[2.30,0.93,0.62,0.74,-0.23],
			[0.93,1.4,0.22,0.56,0.26],
			[0.62,0.22,1.8,0.78,-0.27],
			[0.74,0.56,0.78,3.4,-0.56],
			[-0.23,0.26,-0.27,-0.56,2.6]])
		
		diagG = np.diag(G)
		RiskAsset = [[diagG[i]] for i in range(len(diagG))]
		xdatascale = WindowSize[0]/2
		ydatascale = WindowSize[1]/2
		mu1 = 15.1
		mu2 = 12.5
		mu3 = 14.7
		mu4 = 9.02
		mu5 = 17.68
		ReturnAsset = np.array([[mu1],[mu2],[mu3],[mu4],[mu5]])
		self.add_widget(Label(text="Asset"))
		self.add_widget(Label(text="Risk"))
		self.add_widget(Label(text="Return E[R]"))
		for i in range(len(ReturnAsset)):
			self.add_widget(Label(text=str(i)))
			self.add_widget(Label(text=str(RiskAsset[i])))
			self.add_widget(Label(text=str(ReturnAsset[i])))

		

class TestApp(App):
	def build(self):

		rl = RelativeLayout()
		rl.add_widget(DataWidget())
		rl.add_widget(MyGrid())
		rl.add_widget(LabelWidget())
		return rl

if __name__ == '__main__':
	TestApp().run()
using DelimitedFiles

using Statistics, Optim, SpecialFunctions, LinearAlgebra, LineSearches, Zygote, DataFrames

using IrrationalConstants:twoπ,halfπ,sqrtπ,sqrt2π,invπ,inv2π,invsqrt2,invsqrt2π,logtwo,logπ,log2π

import ChainRulesCore

function MaxCal_Refrac(num_neurons,lambda)

    # Inputs and Outputs

         #=

         Input is the number of neurons (160 for salamander retinal cells) and lambda (can take to be 0.5 for now).

         Here the inputs are (one per neuron, coming from arrays/ folder):
         Time_A (the time intervals of each spike for neuron N)
         Time_Q (the time intervals between each spike for neuron N)
		 State (the state of each neuron at the end of each quiescent period. The state of the neuron of interest N is always 1 here).

        The output Params is an N x N+2 matrix of maximum entropy parameters
				Params[i,j]=Kij, Params[i,i]=hi, Params[i,N+1]=lam_rq_i, Params[i,N+2]=lam_ar_i

         =#

    # Optimization
	#num_neurons=160;
    sz=num_neurons;

    Params=zeros(Float64,sz[1],sz[1]+2);

    for i in 1:sz[1]

        Time_A=readdlm("arrays/Ta_"*string(i, base = 10, pad = 0)*".csv", ',');
		Time_Q=readdlm("arrays/Tqr_"*string(i, base = 10, pad = 0)*".csv", ',');
		State=readdlm("arrays/S_"*string(i, base = 10, pad = 0)*".csv", ',');

        ic=zeros(Float64,1,sz[1]+1);
		ic[end]=1;

        obj(Par)=LogLike(Par,Time_Q,State,lambda,i);
        function g!(G,x)
            G.=obj'(x)
        end

        #optimize step uses automatic differentiation 
        # BFGS estimates second derivative 
        res=optimize(obj,g!, ic,LBFGS(; m=5, linesearch=BackTracking(order=3)));  #slow

        Params[i,1:sz[1]+1]=Optim.minimizer(res);
        Params[i,end]=1/mean(Time_A);

		
		#Compute LogLike(Optim.minimizer(res),Time_Q,State,0,i)
    end

    return Params

end

function LogLike(Param,Time_Q,State,lambda,ii)
    PPP=copy([Param[1:1:ii-1]; Param[ii+1:1:end-1]]);
	b=Param[:,1:end-1]*State;
	b=1/2 .+ 1/2 .* tanh.(b/2);
	p=copy(Param[end]);
	c=b.-p;
	c=expm1.(c.*Time_Q)./c;
	D=length(Time_Q);
	z=sum(log.(c ./ b));

	#return -1. * (D*log(p)+sum(log.(b)) + sum(log.(c)) - sum(b.*Time_Q)) + lambda*norm(PPP)^2;

	return -1. * D*log(p) + z - sum(b.*Time_Q) + lambda*norm(PPP)^2;

end

using DelimitedFiles

using Statistics, Optim, SpecialFunctions, LinearAlgebra, LineSearches, Zygote, DataFrames

using IrrationalConstants:twoπ,halfπ,sqrtπ,sqrt2π,invπ,inv2π,invsqrt2,invsqrt2π,logtwo,logπ,log2π

import ChainRulesCore

include("generate_test_data.jl")

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

    Params=zeros(Float64, 4);

    #temporary
    #Params = [1, 0, 0.5, 0.5]

    i = 1
    print(i)
 
    #concatinates the data
    Time_A=readdlm("salamander/Ta_"*string(i, base = 10, pad = 0)*".csv", ',');
	Time_Q=readdlm("salamander/Tqr_"*string(i, base = 10, pad = 0)*".csv", ',');
	State=readdlm("salamander/S_"*string(i, base = 10, pad = 0)*".csv", ',');
    
    #replace with test data for now
    Time_Q, Time_A = Generate_Test_Data(81000, 0.3, 0.9, 0.7);
    State = ones(1, 81000);

    for i in 2:sz[1]
        print(i)
        Time_A = vcat(Time_A, readdlm("salamander/Ta_"*string(i, base = 10, pad = 0)*".csv", ','));
        Time_Q = vcat(Time_Q, readdlm("salamander/Tqr_"*string(i, base = 10, pad = 0)*".csv", ','));
        State = hcat(State, readdlm("salamander/S_"*string(i, base = 10, pad = 0)*".csv", ','));
    end

    ic=zeros(Float64, 3);
    ic[end]=1;
    #initial guess
    ic = [0.5, 0.0, 0.4]

    obj(Par)=LogLike(Par,Time_Q,State,lambda,i);
    function g!(G,x)
         G.=obj'(x)
    end

    #optimize step uses automatic differentiation 
    # BFGS estimates second derivative 
    res=optimize(obj,g!,  ic,LBFGS(; m=5, linesearch=BackTracking(order=3)));  #slow

    #print(i)

    Params[1:3]=Optim.minimizer(res);
    #print(Optim.minimizer(res))
    Params[end]=1/mean(Time_A);

		
	#Compute LogLike(Optim.minimizer(res),Time_Q,State,0,i)

    return Params

end

# param = [h, J, lambda]
function LogLike(Param,Time_Q,State,lambda,ii)
    b = (tanh.((Param[1] .+ Param[2]*(State' .- 1) )./2).+1)./2;
    g = (tanh(Param[3]/2)+1)/2;
    return -1 .* sum(log.(((g.*b)./(b .- g)).*(exp.(-g.*Time_Q).-exp.(-b.*Time_Q))));
end

print(MaxCal_Refrac(1,0.5))
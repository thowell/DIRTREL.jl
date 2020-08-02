function c_stage!(c,x,u,t)
	nothing
end

function stage_constraints!(c,Z,idx,T,m_stage)
	shift = 0
	for t = 1:T-1
		x = Z[idx.x[t]]
		u = Z[idx.u[t]]
		c_stage!(view(c,shift .+ (1:m_stage[t])),x,u,t)
		shift += m_stage[t]
	end
	nothing
end

function ∇stage_constraints!(∇c,Z,idx,T,m_stage)
	shift = 0
	r_shift = 0
	for t = 1:T-1
		c_tmp = zeros(m_stage[t])

		x = view(Z,idx.x[t])
		u = view(Z,idx.u[t])
		cx(c,z) = c_stage!(c,z,u,t)
		cu(c,z) = c_stage!(c,x,z,t)

		r_idx = r_shift .+ (1:m_stage[t])

		c_idx = idx.x[t]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cx,c_tmp,x))
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(cu,c_tmp,u))
		shift += len

		r_shift += m_stage[t]

	end
	nothing
end

function stage_constraint_sparsity(idx,T,m_stage;shift_r=0,shift_c=0)
	row = []
	col = []
	shift = 0

	for t = 1:T-1
		r_idx = shift_r + shift .+ (1:m_stage[t])

		c_idx = shift_c .+ idx.x[t]
		row_col!(row,col,r_idx,c_idx)

		c_idx = shift_c .+ idx.u[t]
		row_col!(row,col,r_idx,c_idx)

		shift += m_stage[t]
	end
	return collect(zip(row,col))
end

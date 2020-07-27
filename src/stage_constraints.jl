function stage_constraints!(c,Z,idx,T,con,m_con)
	shift = 0
	for t = 2:T-1
		x = Z[idx.x[t]]
		u = Z[idx.u[t]]
		con(view(c,(t-2)*m_con .+ (1:m_con)),x,u)
	end
	nothing
end

function ∇stage_constraints!(∇c,Z,idx,T,con,m_con)
	shift = 0
	c_tmp = zeros(m_con)
	for t = 2:T-1
		x = view(Z,idx.x[t])
		u = view(Z,idx.u[t])
		con_x(c,z) = con(c,z,u)
		con_u(c,z) = con(c,x,z)

		r_idx = (t-2)*m_con .+ (1:m_con)
		c_idx = idx.x[t]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(con_x,c_tmp,x))
		shift += len

		r_idx = (t-2)*m_con .+ (1:m_con)
		c_idx = idx.u[t]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(con_u,c_tmp,u))
		shift += len
	end
	nothing
end

function stage_constraint_sparsity(idx,T,m_con;shift_r=0,shift_c=0)
	row = []
	col = []
	for t = 2:T-1
		r_idx = shift_r + (t-2)*m_con .+ (1:m_con)

		c_idx = shift_c .+ idx.x[t]
		row_col!(row,col,r_idx,c_idx)

		c_idx = shift_c .+ idx.u[t]
		row_col!(row,col,r_idx,c_idx)
	end
	return collect(zip(row,col))
end

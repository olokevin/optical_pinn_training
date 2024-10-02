import torch
import numpy as np
import torch.nn.functional as F
# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize


# Author: Alvin Liu
def quant_func(x,bit=8):
    max_q = 2.0**(bit-1)-1.0
    min_q = -2.0**(bit-1)

    return torch.clip(torch.round(x),min=min_q,max=max_q)


def quant_grad(input,scale=1,bit=8,exp=5,man=2):

    
    if exp==5:
        max_q = 5000
        min_q = -5000
    elif exp==4:
        max_q = 500
        min_q = -500
    else:
        max_q = 1e7
        min_q = -1e7

    max_q = 2.0**(bit-1)-1.0
    min_q = -2.0**(bit-1)
    quant = lambda x: fixed_point_quantize(x,wl=bit,fl=0,rounding="nearest")
    # quant = lambda x: fixed_point_quantize(x,wl=bit,fl=0,rounding="stochastic")

    # quant = lambda x: quant_func(input,bit=bit)

    # quant = lambda x: float_quantize(x, exp=exp, man=man, rounding="nearest")


    min_q = torch.tensor(min_q)
    max_q = torch.tensor(max_q)

    input_div_scale = input/scale
    q_input = quant(input_div_scale)

    grad_scale = (torch.where((input_div_scale<=max_q) & (input_div_scale>=min_q), q_input - input_div_scale, input_div_scale))


    grad_scale = 1e-5*torch.clamp(grad_scale, min = min_q.to(grad_scale.device), max = max_q.to(grad_scale.device))
    return scale*q_input, 0
    # return scale*q_input, torch.sum(grad_scale)


exp_set = 5
man_set = 2

class TT_forward_quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bits,scale_med,scale_grad,scale_y, matrix, *factors):

        # Q = lambda x: quant_grad(x,scale=scale_med,bit=bits,exp=exp_set,man=man_set)
        Q = lambda x: (x,0)

        # print(scale_med)
        # Q = lambda x: (x,0)

        tt_shape = [U.shape[1] for U in factors]
        ndims = len(factors)
        d = int(ndims / 2)

        ctx.input_shape = matrix.shape
        if len(matrix.shape)==3:
            out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
            matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
        else:
            out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
        ctx.out_shape = out_shape

        ctx.bits = bits
        ctx.factors = factors
        ctx.matrix = matrix

        grad_scale = 0
        grad_scale_num = 0
        
   
        ndims = len(factors)
        d = int(ndims / 2)
        ranks = [U.shape[0] for U in factors] + [1]
        tt_shape = [U.shape[1] for U in factors]
        tt_shape_row = list(tt_shape[:d])
        tt_shape_col = list(tt_shape[d:])
        matrix_cols = matrix.shape[0]

        saved_tensors = [matrix]
        left = []
        right = []

        output = factors[0].reshape(-1, ranks[1])
        # print(torch.max(output))
        left.append(output)
        for core in factors[1:d]:
            # print(torch.max(core))
            output_ = torch.tensordot(output, core, dims=([-1], [0]))
            # print(torch.max(output_))


            output,g = Q(torch.tensordot(output, core, dims=([-1], [0])))

            # print(torch.max(output))

            grad_scale += g
            grad_scale_num += np.prod(output.shape)
            left.append(output)

        # output,g = Q(F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0)))
        output,g = Q((matrix@torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0).T))
        grad_scale += g
        grad_scale_num += np.prod(output.shape)

        saved_tensors.append(left)

        temp = factors[d]
        right.append(temp)
        for core in factors[d + 1:]:
            temp,g = Q(torch.tensordot(temp, core, dims=([-1], [0])))
            grad_scale += g
            grad_scale_num += np.prod(temp.shape)
            right.append(temp)
        # print(output.shape)
        # print(torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
        #                                         0, -1).shape)
        # output,g = Q((output@torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
        #                                         0, -1).T).reshape(*out_shape))

        output = (output@torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                                0, -1).T).reshape(*out_shape)
        grad_scale += g
        grad_scale_num += np.prod(output.shape)
        saved_tensors.append(right)

        ctx.saved_tensors_custom = saved_tensors
        ctx.grad_med = grad_scale/grad_scale_num
        ctx.scale_grad = scale_grad
        ctx.scale_y = scale_y

        # print(scale_med)
        # print(grad_scale)

        return output
       
    @staticmethod
    def backward(ctx, dy):
        factors = ctx.factors
        ndims = len(factors)
        d = int(ndims / 2)
        ranks = [U.shape[0] for U in factors] + [1]
        tt_shape = [U.shape[1] for U in factors]
        tt_shape_row = list(tt_shape[:d])
        tt_shape_col = list(tt_shape[d:])
        saved_tensors = ctx.saved_tensors_custom

        # print(dy[0,0,:10])
        # print(dy[0,1,10:20])


        # print('dy=',torch.max(dy))
        # print('dy=',torch.mean(torch.abs(dy)))

        # print('scale_y=',ctx.scale_y.data)
        # print('scale_grad=',ctx.scale_grad.data)

        # scale_y = torch.max(torch.abs(dy))/128
        

        # Q = lambda x: quant_grad(x,scale=ctx.scale_grad,bit=ctx.bits,exp=exp_set,man=man_set)
        Q = lambda x: (x,0)

        # Q = lambda x: quant_grad(x,scale=ctx.scale_grad,bit=8,exp=6,man=3)

        # Q = lambda x: (x,0)
        # if torch.max(dy)>200*ctx.scale_y:
        #     ctx.scale_y.data = 2*ctx.scale_y.data
        # else:
        #     ctx.scale_y.data = 0.5*ctx.scale_y.data
   
        # ctx.scale_y.data[0] = torch.max(dy)/torch.max(dy)*1e-5
        # ctx.scale_y.data[0] = (torch.mean(torch.abs(dy))+0*torch.sqrt(torch.var(torch.abs(dy))))*(1e-1)
        ctx.scale_y.data[0] = (torch.mean(torch.abs(dy))+1*torch.sqrt(torch.var(torch.abs(dy))))*(1e-2)

       

        g_y = 0
        scale_y = ctx.scale_y
        # print('dy=',torch.max(dy))
        # print(torch.norm(dy-quant_grad(dy,scale=scale_y,bit=ctx.bits,exp=exp_set,man=man_set)[0])/torch.norm(dy))
        dy, g_y = quant_grad(dy,scale=scale_y,bit=ctx.bits,exp=exp_set,man=man_set)

        # print(dy.shape)
        # print('>1e-5',torch.sum(torch.abs(dy)>(torch.max(dy)/1000))/torch.prod(torch.tensor(dy.shape)))
        # print('dy=',torch.max(dy))
        # print('dy=',torch.mean(torch.abs(dy)))
        # print('3/4 = ',torch.quantile(torch.abs(dy),9/10))

        # dy[torch.abs(dy)>(torch.max(dy)/1000)] = 0
        # scale_y = 1000*scale_y
        # scale_y = 1
        dy = dy/scale_y
        # print('dy=',torch.max(dy))
        # print('dy=',torch.mean(torch.abs(dy)))

        # g_y = g_y/np.prod(dy.shape)
        g_y = 0

        # print('dy=',torch.max(dy))


        dy = torch.flatten(dy,start_dim=0,end_dim=1)



        g_grad = 0
        g_grad_num = 0

        matrix = saved_tensors[0]
        left = saved_tensors[1]
        right = saved_tensors[2]
        left_grads = []
        right_grads = []

        # with torch.no_grad():
        dy_core_prod = right[-1]
        dy_core_prod,g = Q(torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))
        g_grad+=g
        g_grad_num += np.prod(dy_core_prod.shape)

        matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))

        for i in reversed(range(1, d)):
            # out_ = torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
            #                     matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
            #                                                 ranks[d]),
            #                     dims=([0], [0]))
            # print(torch.max(out_))
            grad,g = Q(torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
                                matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                            ranks[d]),
                                dims=([0], [0])))
            g_grad+=g
            g_grad_num += np.prod(grad.shape)
            if i == d - 1:
                right_core = factors[i]
            else:
                grad,g = Q(torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))
                g_grad+=g
                g_grad_num += np.prod(grad.shape)
                right_core = torch.tensordot(factors[i], right_core,
                                            dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
            
            if grad.shape != factors[i].shape:
                grad = grad.reshape(list(factors[i].shape))

            left_grads.append(grad)
        temp, g = Q(torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                        right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))
        g_grad+=g
        g_grad_num += np.prod(temp.shape)

        left_grads.append(temp)

        left_grads = left_grads[::-1]

        matrix_core_prod = left[-1]
        matrix_core_prod,g = Q(torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                        matrix, dims=([0], [1])))
        g_grad+=g
        g_grad_num += np.prod(matrix_core_prod.shape)
        
        # print('dx=',torch.max(matrix_core_prod))
        matrix_dy_core_prod,g = Q(torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))
        # print('dx=',torch.max(matrix_dy_core_prod))
        g_grad+=g
        g_grad_num += np.prod(matrix_core_prod.shape)

        for i in reversed(range(1, d)):
            grad,g = Q(torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),
                                matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
                                dims=([0], [0])))
            g_grad+=g
            g_grad_num += np.prod(grad.shape)
            if i == d - 1:
                right_core = factors[d + i].reshape(-1, tt_shape_col[i])
            else:
            
                grad,g = Q(torch.tensordot(grad, right_core, dims=([-1], [1])))
                g_grad+=g
                g_grad_num += np.prod(grad.shape)


                right_core,g = Q(torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
                g_grad+=g
                g_grad_num += np.prod(right_core.shape)                                                                                                                                                                     
            if grad.shape != factors[d + i].shape:
                grad = grad.reshape(list(factors[i].shape))

            right_grads.append(grad)

        temp,g = Q(torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                        right_core, dims=([-1], [1])))
        g_grad+=g
        g_grad_num += np.prod(temp.shape)
        right_grads.append(temp)

        right_grads = right_grads[::-1]

        dx = factors[-1].reshape(ranks[-2], -1)
        for core in reversed(factors[d:-1]):
            dx,g = Q(torch.tensordot(core, dx, dims=([-1], [0])))
            g_grad+=g
            g_grad_num += np.prod(dx.shape)

        # print('dx=',torch.max(dx))
        dx,g = Q(torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))
        # print('dx=',torch.max(dx))

        g_grad+=g
        g_grad_num += np.prod(dx.shape)

        temp = factors[0].reshape(-1, ranks[1])
        for core in factors[1:d]:
            temp,g = Q(torch.tensordot(temp, core, dims=([-1], [0])))
            g_grad+=g
            g_grad_num += np.prod(temp.shape)

        dx,g = Q(torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))
        g_grad+=g
        g_grad_num += np.prod(temp.shape)

        dx = torch.reshape(dx,ctx.input_shape)

        # g_grad = 0
        # g_y = 0
        # ctx.grad_med=0
        grad_med = torch.tensor([ctx.grad_med])
        g_grad = torch.tensor([g_grad])/g_grad_num
        g_y = torch.tensor([g_y])

        # print('g_y=',g_y)
        # print('g_grad=',g_grad)
        # print('g_med=',grad_med)
        
        dx = dx*scale_y
        all_grads = [g*scale_y for g in left_grads+right_grads]

        # print(torch.max(all_grads[1]))
        

        return None,grad_med.to(dy.device),g_grad.to(dy.device),g_y.to(dy.device), dx.to(dy.device), *(all_grads)
        # z = torch.tensor(0).to('cuda')
        # [print(U.shape) for U in left_grads+right_grads]
        # return None,z,z,z, dx, *(left_grads + right_grads)



class TT_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):

        with torch.no_grad():

            tt_shape = [U.shape[1] for U in factors]
            ndims = len(factors)
            d = int(ndims / 2)

            ctx.input_shape = matrix.shape
            if len(matrix.shape)==3:
                out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
                matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
            else:
                out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
            ctx.out_shape = out_shape

            ctx.factors = factors
            ctx.matrix = matrix


            
    
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            matrix_cols = matrix.shape[0]

            saved_tensors = [matrix]
            left = []
            right = []

            output = factors[0].reshape(-1, ranks[1])
            left.append(output)

            for core in factors[1:d]:
                output = (torch.tensordot(output, core, dims=([-1], [0])))
                left.append(output)


            output = F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0))


            saved_tensors.append(left)

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                right.append(temp)

            
            # output = (output@torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
            #                                         0, -1).T).reshape(*out_shape)
            
            output = F.linear(output, torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                            0, -1)).reshape(matrix_cols, np.prod(tt_shape_col)).reshape(*out_shape)
        
            
            saved_tensors.append(right)
            ctx.saved_tensors_custom = saved_tensors
       
        # return torch.ones(output.shape).to(output.device)
        return output
       
    @staticmethod
    def backward(ctx, dy):
        with torch.no_grad():
            factors = ctx.factors
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            saved_tensors = ctx.saved_tensors_custom

            
            


            if len(dy.shape)==3:
                dy = torch.flatten(dy,start_dim=0,end_dim=1)
            # dy = torch.mean(dy,dim=0).unsqueeze(0)







            matrix = saved_tensors[0]
            left = saved_tensors[1]
            right = saved_tensors[2]
            left_grads = []
            right_grads = []

            # with torch.no_grad():
            dy_core_prod = right[-1]

            # print(dy.shape)
            # print( dy_core_prod.reshape(dy_core_prod.shape[0], -1).shape)
        
            dy_core_prod = (torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))
            # print(dy_core_prod.shape)

            matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))
            # print(matrix_dy_core_prod.shape)

            for i in reversed(range(1, d)):
                grad = (torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
                                    matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                                ranks[d]),
                                    dims=([0], [0])))
                # print(grad.shape)
                if i == d - 1:
                    right_core = factors[i]
                else:
                    grad = (torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))

                    right_core = torch.tensordot(factors[i], right_core,
                                                dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
                
                if grad.shape != factors[i].shape:
                    grad = grad.reshape(list(factors[i].shape))
                # print(grad.shape)
                left_grads.append(grad)
            temp = (torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                            right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))


            left_grads.append(temp)

            left_grads = left_grads[::-1]

            matrix_core_prod = left[-1]
            matrix_core_prod = (torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                            matrix, dims=([0], [1])))

            
            # print('dx=',torch.max(matrix_core_prod))
            matrix_dy_core_prod = (torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))


            for i in reversed(range(1, d)):
                grad = (torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),
                                    matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
                                    dims=([0], [0])))
            
                if i == d - 1:
                    right_core = factors[d + i].reshape(-1, tt_shape_col[i])
                else:
                
                    grad = (torch.tensordot(grad, right_core, dims=([-1], [1])))
                    


                    right_core = (torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
                                                                                                                                                                            
                if grad.shape != factors[d + i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                right_grads.append(grad)

            temp = (torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                            right_core, dims=([-1], [1])))

            right_grads.append(temp)

            right_grads = right_grads[::-1]

            dx = factors[-1].reshape(ranks[-2], -1)
            for core in reversed(factors[d:-1]):
                dx = (torch.tensordot(core, dx, dims=([-1], [0])))

        
            dx = (torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))



            temp = factors[0].reshape(-1, ranks[1])
            for core in factors[1:d]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))


            dx = (torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))

            dx = torch.reshape(dx,ctx.input_shape)
            
            # n = len(ctx.input_shape) - 1
            # dx = torch.reshape(dx,ctx.input_shape[n:])
            # if i in range(n):
            #     dx = dx.unsqueeze(0)
            # dx = dx.repeat(list(ctx.input_shape[0:n])+[1]*(len(ctx.input_shape)-n))

            

            all_grads = [g for g in left_grads+right_grads]


        # return None,None
        return dx, *(all_grads)
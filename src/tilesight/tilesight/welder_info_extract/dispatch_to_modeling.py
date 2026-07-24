from tilesight.fused_op_dtype import Operation
from tilesight.fused_op_dtype import *
from tilesight.welder_info_extract.welder_config import Welder_Config
from tilesight.arch import *
from tilesight.compare_with_ncu import *
from tilesight.fusion_support import *
import math


# # def handle_nn_conv2d_nn_bias_add(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
# #     # TODO
# #     pass

# def handle_nn_conv2d(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="nn.conv2d"
#     # padding=[1, 1, 1, 1], groups=64, channels=64, kernel_size=[3, 3]
#     single_op=Operation(single_op_name)
#     pass

# def handle_subtract(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
    
#     single_op=Operation(single_op_name)
    
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass


# def handle_multiply(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     single_op=Operation(single_op_name)
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_mean(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op=Operation(single_op_name)
     
#     # axis=[1], keepdims=True
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_add(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     single_op=Operation(single_op_name)
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_sqrt(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     single_op=Operation(single_op_name)
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_divide(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     single_op=Operation(single_op_name)
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_welder_C2DImplicitGemm(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="welder.C2DImplicitGemm"
#     single_op=Operation(single_op_name)
     
#     # welder.C2DImplicitGemm(%p026, %p121, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]) /* ty=Tensor[(64, 1048576), float16] */
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

# def handle_nn_bias_add(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="nn.bias_add"
#     single_op=Operation(single_op_name)
     
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

#     pass

# # def handle_reshape_transpose_reshape(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
# #     # TODO
# #     single_op_name="reshape"
# #     single_op=Operation(single_op_name)
# #      
# #     # print("new_line",new_line)
# #     # print("single_op",single_op)
# #     pass

# #     pass
# def handle_reshape(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="reshape"
#     single_op=Operation(single_op_name)
     
#     # %40 = reshape(%p022, newshape=[32, 64, 128, 128]) /* ty=Tensor[(32, 64, 128, 128), float16] */;
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

#     pass

# def handle_transpose(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="transpose"
#     single_op=Operation(single_op_name)
     
#     # axes=[1, 0, 2, 3]
#     # print("new_line",new_line)
#     # print("single_op",single_op)
#     pass

#     pass

# def handle_strided_slice(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     # %26 = strided_slice(%p014, begin=[0i64], end=[16i64], strides=[1i64], axes=[1i64])
#     pass

# def handle_nn_global_avg_pool2d(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     # TODO
#     single_op_name="nn.global_avg_pool2d"
#     # nn.global_avg_pool2d(%p0113) /* ty=Tensor[(64, 32, 1, 1), float16] span=/encoders.1/encoders.1.0/sca/sca.0/GlobalAveragePool:0:0 */
#     pass

# def handle_nn_depth_to_space(single_op_name,reg_fused_nodes,current_relay,current_line,row_panel):
#     single_op_name="nn.depth_to_space"
#     # %112 = reshape(%111, newshape=[64, 256, 32, 32]) /* ty=Tensor[(64, 256, 32, 32), float16] */;
#     # %113 = nn.depth_to_space(%112, block_size=2, mode="CRD") /* ty=Tensor[(64, 64, 64, 64), float16] span=/ups.0/ups.0.1/DepthToSpace:0:0 */;
#     # TODO
#     pass

# # 创建操作名称到处理函数的映射
# op_handlers = {
#     # "nn_conv2d_nn_bias_add": handle_nn_conv2d_nn_bias_add,
#     # "reshape_transpose_reshape": handle_reshape_transpose_reshape,
#     "reshape": handle_reshape,
#     "transpose": handle_transpose,
#     "nn_conv2d": handle_nn_conv2d,
#     "nn_bias_add": handle_nn_bias_add,
#     "subtract": handle_subtract,
#     "multiply": handle_multiply,
#     "mean": handle_mean,
#     "add": handle_add,
#     "sqrt": handle_sqrt,
#     "divide": handle_divide,
#     "welder_C2DImplicitGemm": handle_welder_C2DImplicitGemm,
#     "strided_slice": handle_strided_slice,
#     "nn_global_avg_pool2d": handle_nn_global_avg_pool2d,
#     "nn_depth_to_space": handle_nn_depth_to_space,
#     # 其他操作处理函数
# }


def closest_divisor_to_sqrt(n):
    # 获取输入数的平方根
    sqrt_n = int(math.sqrt(n))
    
    # 从平方根开始，向下寻找第一个可以整除n的数
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return i

def model_element_wise(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    # print("op.op_name",op.op_name)
    # print("op.input1_shape",op.input1_shape)
    # print("op.input2_shape",op.input2_shape)

    # if(op.input2_shape==[] or op.input2_shape==[1]):
    if(op.input2_shape==[]):
        return model_element_wise_N_0(op,welder_config,arch_spec)
    elif (len(op.input1_shape)!=len(op.input2_shape)):
        # print("len(op.input2_shape)!=len(op.input2_shape)")

        if (len(op.input1_shape)<len(op.input2_shape)):
            op.exchange_input1_input2()
        
        if(op.op_name=="nn.bias_add"):
            if(len(op.input2_shape)!=1):
                raise ValueError(f"Error in nn.bias_add kernel of {op.op_name},{op.input2_shape}")
            op.input2_shape=[1,op.input1_shape[1],1,1]
        else:
            # insert len(op.input1_shape)-len(op.input2_shape) 's one
            while len(op.input2_shape) < len(op.input1_shape):
                op.input2_shape.insert(0, 1)
        # After making sure lengths are matched, check dimension compatibility
        # print("op.name",op.op_name)
        # print("op.input1_shape",op.input1_shape,"op.input2_shape",op.input2_shape)
        for i in range(len(op.input2_shape)):
            if op.input2_shape[i] != op.input1_shape[i] and op.input2_shape[i] != 1:
                # print("op.name",op.op_name)
                # print("op.input1_shape",op.input1_shape,"op.input2_shape",op.input2_shape)
                raise ValueError(f"Error of input1 and input2 mismatch at {op.op_name}'s {op.input1_shape} {op.input2_shape}")
        
        # print("op.input1_shape",op.input1_shape)
        # print("op.input2_shape",op.input2_shape)
        return model_element_wise_N_1(op,welder_config,arch_spec)
    
    elif (op.input2_shape!=op.input1_shape):
        input1_shape=op.input1_shape
        input2_shape=op.input2_shape
        for i in range(len(op.input1_shape)):
            if op.input1_shape[i]<op.input2_shape[i]:
                op.input1_shape=input2_shape
                op.input2_shape=input1_shape

                break
        # print("op.input1_shape",op.input1_shape,"op.input2_shape",op.input2_shape)
        return model_element_wise_N_1(op,welder_config,arch_spec)
    elif (op.input2_shape == op.input1_shape):
        return model_element_wise_N_N(op,welder_config,arch_spec)
        


    pass

def model_element_wise_N_0(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    
    mem_levels = op.get_mem_levels()

    op_shape=op.output_shape
    

    tb_shape,dim_threads=welder_config.get_block_thread(op_shape)

    

    ret=calculate_N_0_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch_spec,mem_levels,num_ops=1)
    # calculate_N_0_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1)
    # return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io

    post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)

    return post_data

def model_element_wise_N_0_sfu(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    
    mem_levels = op.get_mem_levels()

    op_shape=op.output_shape
    

    tb_shape,dim_threads=welder_config.get_block_thread(op_shape)

    

    ret=calculate_N_0_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch_spec,mem_levels,num_ops=1)
    # calculate_N_0_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1)
    # return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io

    post_data=hyper_post_process_sfu_core_op(ret,arch_spec)

    return post_data

def model_element_wise_N_1(op:Operation,welder_config:Welder_Config,arch_spec:Arch):

    mem_levels = op.get_mem_levels()

    op_shape=op.output_shape
    in2_shape=op.input2_shape

    tb_shape,dim_threads=welder_config.get_block_thread(op_shape)

    ret=calculate_N_1_elementwise_resource_utilization(op_shape,in2_shape,tb_shape,dim_threads, arch_spec,mem_levels,num_ops=1)
    post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)

    return post_data

def model_element_wise_N_N(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    mem_levels = op.get_mem_levels()
    op_shape=op.output_shape
    tb_shape,dim_threads=welder_config.get_block_thread(op_shape)

    ret=calculate_N_N_elementwise_resource_utilization(op_shape,tb_shape,dim_threads, arch_spec,mem_levels,num_ops=1)

    post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)

    return post_data

def model_conv_implicit_gemm(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    # TODO: NEED FURTHER UPDATE ON CONV IMPLICIT GEMM MODELING FUNCTION!
    kernel_size=op.op_special_metric_dict['kernel_size']
    strides = op.op_special_metric_dict.get('strides')
    channels=op.op_special_metric_dict['channels']
    padding=op.op_special_metric_dict['padding']
    if 'data_layout' in op.op_special_metric_dict:
        data_layout = op.op_special_metric_dict['data_layout']
        if data_layout == 'NHWC':
            # axis 0->0, 1->2, 2->3, 3->1
            op.input1_shape = [op.input1_shape[i] for i in [0, 3, 1, 2]]
            # print("op.input1_shape = [op.input1_shape[i] for i in [0, 3, 1, 2]]",op.input1_shape)
            pass
    if 'kernel_layout' in op.op_special_metric_dict:
        kernel_layout = op.op_special_metric_dict['kernel_layout']
    
    mem_levels = op.get_mem_levels()

    if welder_config.use_tc!=None:
        

        # out_shape=op.output_shape

        

        conv_n=op.input1_shape[0]
        conv_f=op.input2_shape[1]
        

        if strides is None:
            conv_h=op.input1_shape[2]
            conv_w=op.input1_shape[3]
            conv_c=op.input1_shape[1]
            conv_kh=kernel_size[0]
            conv_kw=kernel_size[1]
            # 如果strides是None，执行这部分代码
            # print("strides键不存在或其值为None")
        else:
            conv_h=op.input1_shape[2]//strides[0]
            conv_w=op.input1_shape[3]//strides[1]
            # conv_c=op.input1_shape[1]*strides[0]*strides[1]
            conv_c=op.input1_shape[1]
            conv_kh=kernel_size[0]+padding[0]
            conv_kw=kernel_size[1]+padding[2]

        tb_m=welder_config.block[0]
        tb_n=welder_config.block[1]
        tb_k=welder_config.rstep[0]

        wp_m=welder_config.warp[0]
        wp_n=welder_config.warp[1]
        wp_k=tb_k
            
        op_shape=(conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw)
        tb_shape=(tb_m,tb_n,tb_k)
        wp_shape=(wp_m,wp_n,wp_k)
        
        bytes_per_num=op.output_bytes
        stage_num=2
        arch=arch_spec
            # 如果strides不是None，执行这部分代码
            # print(f"strides的值为: {strides}")    

        # conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw=op_shape
        # tb_m,tb_n,tb_k=tb_shape
        # wp_m,wp_n,wp_k=wp_shape
        ret=calculate_conv_implicit_gemm_resource_utilization(op_shape,tb_shape,wp_shape, bytes_per_num, stage_num,arch,mem_levels)
        # print("ret",ret)
        # print("(conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw)",(conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw))
        post_data=hyper_post_process_tensor_core_op(ret,arch_spec,op.input1_bytes)
        # print("Time",post_data)
        
    else :
        arch=arch_spec
        if welder_config.reduce_thread!=[]:
            conv_n=op.input1_shape[0]
            conv_f=op.input2_shape[1]
            conv_h=op.input1_shape[2]
            conv_w=op.input1_shape[3]
            conv_c=op.input1_shape[1]
            conv_kh=kernel_size[0]
            conv_kw=kernel_size[1]
            
            if strides is None:
                conv_h=op.input1_shape[2]
                conv_w=op.input1_shape[3]
                conv_c=op.input1_shape[1]
                conv_kh=kernel_size[0]
                conv_kw=kernel_size[1]
                # 如果strides是None，执行这部分代码
                # print("strides键不存在或其值为None")
            else:
                conv_h=op.input1_shape[2]//strides[0]
                conv_w=op.input1_shape[3]//strides[1]
                # conv_c=op.input1_shape[1]*strides[0]*strides[1]
                conv_c=op.input1_shape[1]
                conv_kh=kernel_size[0]
                conv_kw=kernel_size[1]        

            # out_shape=(conv_n,conv_f,conv_h,conv_w)
            out_shape=(conv_n*conv_h*conv_w,conv_f)
            if(conv_kh==1 and conv_kw==1):
                reduction_shape=[conv_c]
                reduction_axis_mapping=[[2,welder_config.rstep[0]]]

            else:
                # reduction_shape=[conv_c,conv_kh,conv_kw]
                # reduction_axis_mapping=[[2,welder_config.rstep[0]],[1,welder_config.rstep[1]],[1,welder_config.rstep[2]]]
                reduction_shape=[conv_c*conv_kh*conv_kw]
                reduction_axis_mapping=[[2,welder_config.rstep[0]]]

            # out_axis_mapping=[0,1,0,0]
            out_axis_mapping=[0,1]
            out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)

            reduce_threads_info=[1,1]

            for idx in range(len(welder_config.reduce_thread)):
                if welder_config.reduce_thread[idx]!=1:
                    reduce_threads_info=[idx,welder_config.reduce_thread[idx]]

            # ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=1)
            ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=0)
            post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)

        else:
            conv_n=op.input1_shape[0]
            conv_f=op.input2_shape[1]
            conv_h=op.input1_shape[2]
            conv_w=op.input1_shape[3]
            conv_c=op.input1_shape[1]
            conv_kh=kernel_size[0]
            conv_kw=kernel_size[1]

            if strides is None:
                conv_h=op.input1_shape[2]
                conv_w=op.input1_shape[3]
                conv_c=op.input1_shape[1]
                conv_kh=kernel_size[0]
                conv_kw=kernel_size[1]
                # 如果strides是None，执行这部分代码
                # print("strides键不存在或其值为None")
            else:
                conv_h=op.input1_shape[2]//strides[0]
                conv_w=op.input1_shape[3]//strides[1]
                # conv_c=op.input1_shape[1]*strides[0]*strides[1]
                conv_c=op.input1_shape[1]
                conv_kh=kernel_size[0]
                conv_kw=kernel_size[1]           

            # out_shape=(conv_n,conv_f,conv_h,conv_w)
            out_shape=(conv_n*conv_h*conv_w,conv_f)
            # print("conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw",conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw)
            if(conv_kh==1 and conv_kw==1):
                reduction_shape=[conv_c]
                reduction_axis_mapping=[[2,welder_config.rstep[0]]]

            else:
                # reduction_shape=[conv_c,conv_kh,conv_kw]
                # reduction_axis_mapping=[[2,welder_config.rstep[0]],[1,welder_config.rstep[1]],[1,welder_config.rstep[2]]]
                reduction_shape=[conv_c*conv_kh*conv_kw]
                reduction_axis_mapping=[[2,welder_config.rstep[0]]]

            out_axis_mapping=[0,1]
            out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)

            # ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1)
            ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=0)
            post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
            pass
    
    return post_data
    # print("post_data",post_data)
    

def model_conv_2d(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    kernel_size=op.op_special_metric_dict['kernel_size']
    padding = op.op_special_metric_dict.get('padding')
    groups = op.op_special_metric_dict.get('groups')
    channel = op.op_special_metric_dict.get('channel')
    strides = op.op_special_metric_dict.get('strides')
    # print("op",op)
    # print("welder_config",welder_config)
    # print("groups",groups)
    mem_levels=op.get_mem_levels()
    if groups ==None:
        if welder_config.reduce_thread==[]:
            if strides is None:
                out_shape=op.output_shape
                reduction_shape=[op.input1_shape[1],kernel_size[0],kernel_size[1]]
                out_axis_mapping=[0,1,0,0]
                reduction_axis_mapping=[
                        [2, welder_config.rstep[0]],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                        [1, welder_config.rstep[1]],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                        [1, welder_config.rstep[2]],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                ]
                
                out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)

                #     n, f, h, w, c, kh, kw, s, d, p = 64, 16, 256, 256, 3, 3, 3, 1, 1, 1
                #     # 2023-12-07 11:25:14 [welder:DEBUG]: Best Config: {'globals': {'Rasterization': <NoRasterization>}, <Node, nn_conv2d_nn_bias_add_0>: {'block': [1, 16, 4, 64], 'thread': [1, 8, 1, 16], 'rstep': [3, 3, 3]}}
                #     # 2023-12-07 11:25:14 [welder:INFO]: result: 0.32040369510650635
                #     conv_levels = {
                #             'in1': [1,1,1], 
                #             'in2': [1,1,1],
                #             'out1': [0,0,1], }
                #     out_axis_mapping=[0,1,0,0]
                #     reduction_axis_mapping=[
                #         [2, 3],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                #         [1, 3],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                #         [1, 3],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                # ]
                #     reg_array=[]
                ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1)
                # ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=0)
                post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
            else:
                out_shape=op.output_shape
                # op.input1_shape[2]=op.input1_shape[2]/strides[0]
                # op.input1_shape[3]=op.input1_shape[3]/strides[1]
                reduction_shape=[op.input1_shape[1],kernel_size[0],kernel_size[1]]
                out_axis_mapping=[0,1,0,0]
                reduction_axis_mapping=[
                        [2, welder_config.rstep[0]],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                        [1, welder_config.rstep[1]],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                        [1, welder_config.rstep[2]],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                ]
                
                out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)

                #     n, f, h, w, c, kh, kw, s, d, p = 64, 16, 256, 256, 3, 3, 3, 1, 1, 1
                #     # 2023-12-07 11:25:14 [welder:DEBUG]: Best Config: {'globals': {'Rasterization': <NoRasterization>}, <Node, nn_conv2d_nn_bias_add_0>: {'block': [1, 16, 4, 64], 'thread': [1, 8, 1, 16], 'rstep': [3, 3, 3]}}
                #     # 2023-12-07 11:25:14 [welder:INFO]: result: 0.32040369510650635
                #     conv_levels = {
                #             'in1': [1,1,1], 
                #             'in2': [1,1,1],
                #             'out1': [0,0,1], }
                #     out_axis_mapping=[0,1,0,0]
                #     reduction_axis_mapping=[
                #         [2, 3],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                #         [1, 3],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                #         [1, 3],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                # ]
                #     reg_array=[]
                # ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1)
                ret=calculate_general_ruduce_resource_utilization_with_stride(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1,strides=strides)
                post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
        elif welder_config.reduce_thread!=[]:
            if strides is None:
                out_shape=op.output_shape
                reduction_shape=[op.input1_shape[1],kernel_size[0],kernel_size[1]]
                out_axis_mapping=[0,1,0,0]
                reduction_axis_mapping=[
                        [2, welder_config.rstep[0]],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                        [1, welder_config.rstep[1]],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                        [1, welder_config.rstep[2]],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                ]
                
                out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)
                reduce_thread=welder_config.reduce_thread
                for tmp_index in [0,1,2]:
                    if reduce_thread[tmp_index]!=1:
                        reduce_threads_info=[tmp_index,reduce_thread[tmp_index]]
                        break
                # reduce_threads_info=[0,2] 
                # reduce thread at axis 0, reduce thread is 2

                #     n, f, h, w, c, kh, kw, s, d, p = 64, 16, 256, 256, 3, 3, 3, 1, 1, 1
                #     # 2023-12-07 11:25:14 [welder:DEBUG]: Best Config: {'globals': {'Rasterization': <NoRasterization>}, <Node, nn_conv2d_nn_bias_add_0>: {'block': [1, 16, 4, 64], 'thread': [1, 8, 1, 16], 'rstep': [3, 3, 3]}}
                #     # 2023-12-07 11:25:14 [welder:INFO]: result: 0.32040369510650635
                #     conv_levels = {
                #             'in1': [1,1,1], 
                #             'in2': [1,1,1],
                #             'out1': [0,0,1], }
                #     out_axis_mapping=[0,1,0,0]
                #     reduction_axis_mapping=[
                #         [2, 3],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                #         [1, 3],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                #         [1, 3],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                # ]
                #     reg_array=[]
                # ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1)
                ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch_spec,mem_levels,compute_at=1)
                # ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch_spec,mem_levels,compute_at=0)
                post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
            else:
                out_shape=op.output_shape
                op.input1_shape[2]=op.input1_shape[2]/strides[0]
                op.input1_shape[3]=op.input1_shape[3]/strides[1]
                reduction_shape=[op.input1_shape[1],kernel_size[0],kernel_size[1]]
                out_axis_mapping=[0,1,0,0]
                reduction_axis_mapping=[
                        [2, welder_config.rstep[0]],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                        [1, welder_config.rstep[1]],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                        [1, welder_config.rstep[2]],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                ]
                
                out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)
                reduce_thread=welder_config.reduce_thread
                for tmp_index in [0,1,2]:
                    if reduce_thread[tmp_index]!=1:
                        reduce_threads_info=[tmp_index,reduce_thread[tmp_index]]
                        break

                #     n, f, h, w, c, kh, kw, s, d, p = 64, 16, 256, 256, 3, 3, 3, 1, 1, 1
                #     # 2023-12-07 11:25:14 [welder:DEBUG]: Best Config: {'globals': {'Rasterization': <NoRasterization>}, <Node, nn_conv2d_nn_bias_add_0>: {'block': [1, 16, 4, 64], 'thread': [1, 8, 1, 16], 'rstep': [3, 3, 3]}}
                #     # 2023-12-07 11:25:14 [welder:INFO]: result: 0.32040369510650635
                #     conv_levels = {
                #             'in1': [1,1,1], 
                #             'in2': [1,1,1],
                #             'out1': [0,0,1], }
                #     out_axis_mapping=[0,1,0,0]
                #     reduction_axis_mapping=[
                #         [2, 3],  # 规约轴的第一个'c' 是公共归约轴，current_step是8
                #         [1, 3],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                #         [1, 3],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                # ]
                #     reg_array=[]
                # ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=1)
                ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch_spec,mem_levels,compute_at=1)
                # ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch_spec,mem_levels,compute_at=0)
                post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)

   
    elif groups !=None:
        if welder_config.reduce_thread==[]:
            if strides is None:
                out_shape=op.output_shape
                reduction_shape=[kernel_size[0],kernel_size[1]]
                out_axis_mapping=[0,1,0,0]
                reduction_axis_mapping=[
                        [1, welder_config.rstep[0]],  # 规约轴的第二个'kh' 是input2的私有归约轴，current_step是3
                        [1, welder_config.rstep[1]],  # 规约轴的第三个'kw' 是input2的私有规约轴，current_step是3
                ]
                
                out_tb_shape,dim_threads=welder_config.get_block_thread(out_shape)
                
                # ret=calculate_general_ruduce_resource_utilization([n,f,h,w],[kh,kw],out_axis_mapping,reduction_axis_mapping,[1, 32, 16, 16],[1, 32, 2, 2],4, chip,dw_conv_levels,compute_at=0)
                ret=calculate_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch_spec,mem_levels,compute_at=0)
                post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
                pass


    return post_data



def model_reduce(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    # print(welder_config)
    if (op.output_shape==op.input1_shape):
        return model_element_wise_N_0(op,welder_config,arch_spec)

    if welder_config.reduce_thread == []:
        return model_reduce_no_reduce_thread(op,welder_config,arch_spec)
    else:
        return model_reduce_with_reduce_thread(op,welder_config,arch_spec)



    pass

def model_reduce_no_reduce_thread(op:Operation,welder_config:Welder_Config,arch_spec:Arch):

    if op.input2_shape==[] or op.input2_shape ==[1]:
        out_shape=op.output_shape
        reduction_shape=[]
        reduction_axis_mapping=[]
        out_axis_mapping = [0 for _ in range(len(op.output_shape))]
        out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)
        arch=arch_spec
        mem_levels=op.get_mem_levels()
        
        for idx in range(len(op.output_shape)):
            if op.output_shape[idx]!= op.input1_shape[idx]:
                reduction_shape.append(op.input1_shape[idx]//op.output_shape[idx])

        for idx in range(len(reduction_shape)):
            reduction_axis_mapping.append([0,welder_config.rstep[idx]])

        # print("out_shape",out_shape)
        # print("out_axis_mapping",out_axis_mapping)
        # print("reduction_shape",reduction_shape)
        # print("reduction_axis_mapping",reduction_axis_mapping)
        # print("out_tb_shape",out_tb_shape)
        # print("dim_threads",dim_threads)
        # print("op",op)
        # print("welder_config",welder_config)

        ret=calculate_N_0_general_ruduce_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch,mem_levels,compute_at=1)
        post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
    # if op.input2_shape==[]:
        #only one op

    return post_data
    pass

def model_reduce_with_reduce_thread(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    
    if op.input2_shape==[] or op.input2_shape==[1]:
        out_shape=op.output_shape
        reduction_shape=[]
        reduction_axis_mapping=[]
        out_axis_mapping = [0 for _ in range(len(op.output_shape))]
        out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)
        arch=arch_spec
        mem_levels=op.get_mem_levels()
        reduce_threads_info=[1,1]
        
        for idx in range(len(op.output_shape)):
            if op.output_shape[idx]!= op.input1_shape[idx]:
                reduction_shape.append(op.input1_shape[idx]//op.output_shape[idx])

        for idx in range(len(reduction_shape)):
            reduction_axis_mapping.append([0,welder_config.rstep[idx]])

        for idx in range(len(welder_config.reduce_thread)):
            if welder_config.reduce_thread[idx]!=1:
                reduce_threads_info=[idx,welder_config.reduce_thread[idx]]
        # print("out_shape",out_shape)
        # print("out_axis_mapping",out_axis_mapping)
        # print("reduction_shape",reduction_shape)
        # print("reduction_axis_mapping",reduction_axis_mapping)
        ret=calculate_N_0_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=1)
        print(ret)
        post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
        # print("boundtime",post_data[0])
    # if op.input2_shape==[]:
        #only one op
    else: 
        raise ValueError(f"No handler found for operation: {op.op_name}")
    

    return post_data

def model_gemm(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    
    if (len(op.output_shape)==2):
        m=op.output_shape[0]
        n=op.output_shape[1]
        k=op.input1_shape[1]
        # {'block': [16, 8, 16, 16], 'warp': [8, 4, 16, 16], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32, 1], 'use_tc': '80', 'strides': {2: <Stride, 2, 16>}}}
        # 2024-06-19 19:07:23 [ladder:INFO]: result: 20.159282684326172
        out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)

        bm=out_tb_shape[0]
        bn=out_tb_shape[1]
        bk=welder_config.rstep[0]

        wp_m=welder_config.warp[0]
        wp_n=welder_config.warp[1]
        wp_k=welder_config.rstep[0]

        mem_levels=op.get_mem_levels()
        row_panel=welder_config.row_panel

        ret=calculate_ladder_matmul_resource_utilization([m,n,k],[bm,bn,bk],[wp_m,wp_n,wp_k],2,arch_spec,mem_levels,row_panel,batch=1,split_param=1)
        post_data=hyper_post_process_tensor_core_op(ret,arch_spec, op.input1_bytes)

    elif(len(op.output_shape)==3):  
        bs=op.output_shape[0]
        m=op.output_shape[1]
        n=op.output_shape[2]
        k=op.input1_shape[2]
        # {'block': [16, 8, 16, 16], 'warp': [8, 4, 16, 16], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32, 1], 'use_tc': '80', 'strides': {2: <Stride, 2, 16>}}}
        # 2024-06-19 19:07:23 [ladder:INFO]: result: 20.159282684326172
        out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)

        bm=out_tb_shape[1]
        bn=out_tb_shape[2]
        bk=welder_config.rstep[0]

        wp_m=welder_config.warp[1]
        wp_n=welder_config.warp[2]
        wp_k=welder_config.rstep[0]

        mem_levels=op.get_mem_levels()
        row_panel=welder_config.row_panel

        ret=calculate_ladder_matmul_resource_utilization([m,n,k],[bm,bn,bk],[wp_m,wp_n,wp_k],2,arch_spec,mem_levels,row_panel,batch=bs,split_param=1)
        post_data=hyper_post_process_tensor_core_op(ret,arch_spec, op.input1_bytes)
        pass
    else:
        raise ValueError(f"No handler found for operation: {op.op_name}")
    
    return post_data

def model_ladder_matmul(op,welder_config,arch_spec):
    # "config": "{'globals': {'Rasterization': <NoRasterization>}, <Node, ladder_perfect_matmul_cast_1__layout_transform_reshape_2>: 
    # {'block': [8, 8, 16, 16], 'warp': [4, 4, 16, 16], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32, 1], 
    # 'use_tc': 'gfx90a', 'strides': {2: <Stride, 2, 16>}}}",
    m=op.input1_shape[0]*op.input1_shape[2]
    k=op.input1_shape[1]*op.input1_shape[3]
    n=op.input2_shape[0]*op.input2_shape[2]
    # print("m,k,n",m,k,n)
    out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)
    # out_tb_shape [8, 8, 16, 16] dim_threads [5, 3, 3, 3]
    # print("out_tb_shape",out_tb_shape,"dim_threads",dim_threads)
    bm=out_tb_shape[0]*out_tb_shape[2]
    bn=out_tb_shape[1]*out_tb_shape[3]
    bk=welder_config.rstep[0]
    # print("bm,bn,bk",bm,bn,bk)
    wp_m=welder_config.warp[0]*welder_config.warp[2]
    wp_n=welder_config.warp[1]*welder_config.warp[3]
    wp_k=welder_config.rstep[0]
    # print("wp_m,wp_n,wp_k",wp_m,wp_n,wp_k)
    mem_levels=op.get_mem_levels()
    # row_panel=welder_config.row_panel
    row_panel= closest_divisor_to_sqrt(arch_spec.sm_count)
    # print("mem_levels",mem_levels,"row_panel",row_panel)
    ret=calculate_ladder_matmul_resource_utilization([m,n,k],[bm,bn,bk],[wp_m,wp_n,wp_k],2,arch_spec,mem_levels,row_panel,batch=1,split_param=1)
    post_data=hyper_post_process_tensor_core_op(ret,arch_spec, op.input1_bytes)    
    
    return post_data
    

    pass

def model_ladder_batch_matmul(op,welder_config,arch_spec):
        # "config": "{'globals': {'Rasterization': <NoRasterization>}, <Node, ladder_perfect_matmul_cast_1__layout_transform_reshape_2>: 
    # {'block': [8, 8, 16, 16], 'warp': [4, 4, 16, 16], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32, 1], 
    # 'use_tc': 'gfx90a', 'strides': {2: <Stride, 2, 16>}}}",
    # print("welder_config",welder_config)

    mem_levels=op.get_mem_levels()
    
    if (welder_config.wmma==[] and len(op.output_shape)==3):
        # print("op",op)
        # print("welder_config",welder_config)

        bs=op.output_shape[0]
        m=op.output_shape[1]
        n=op.output_shape[2]
        k=op.input1_shape[2]


        out_shape=op.output_shape
        reduction_shape=[k]
        out_axis_mapping=[2,0,1]
        reduction_axis_mapping=[[2,welder_config.rstep[0]]]
        out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)
        
        reduce_thread= 1 if welder_config.reduce_thread==[] else welder_config.reduce_thread[0]
        reduce_threads_info=[0,reduce_thread]
        
        # sadjiojiow:;


        #  <Node, nn_batch_matmul_9>: {'block': [8, 1, 1], 'thread': [8, 1, 1], 'rstep': [128], 'reduce_thread': [16], 'vectorize': {'p0': 8, 'p1': 8}}}",
        # reg_array=[]

        # matmul_levels={'in1': [0,1,1,2], 'in2': [1,1,1,2],'out1': [1,1,1,2]}
        # out_axis_mapping=[2,2,0,1]
        # reduction_axis_mapping=[[2,128]]
        # reduce_threads_info=[0,16]
        # ret=calculate_general_ruduce_inter_thread_resource_utilization([bs,num_attention_heads,seq,seq],[hidden_size/num_attention_heads],out_axis_mapping,reduction_axis_mapping
        #                                                                ,[8,1,1,1],[8,1,1,1],reduce_threads_info,chip,matmul_levels,compute_at=1)
        # ret=calculate_ladder_matmul_resource_utilization([m,n,k],[bm,bn,bk],[wp_m,wp_n,wp_k],2,arch_spec,mem_levels,row_panel,batch=1,split_param=1)
        
        ret=calculate_general_ruduce_inter_thread_resource_utilization(out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch_spec,mem_levels,compute_at=1)
        post_data=hyper_post_process_cuda_core_op(ret,arch_spec, op.input1_bytes)
        return post_data

    bs=op.output_shape[0]
    m=op.output_shape[1]
    n=op.output_shape[2]
    k=op.input1_shape[2]
    # {'block': [16, 8, 16, 16], 'warp': [8, 4, 16, 16], 'wmma': [16, 16, 16], 'use_cutlass': False, 'rstep': [32, 1], 'use_tc': '80', 'strides': {2: <Stride, 2, 16>}}}
    # 2024-06-19 19:07:23 [ladder:INFO]: result: 20.159282684326172
    out_tb_shape,dim_threads=welder_config.get_block_thread(op.output_shape)

    bm=out_tb_shape[1]
    bn=out_tb_shape[2]
    bk=welder_config.rstep[0]

    wp_m=welder_config.warp[1]
    wp_n=welder_config.warp[2]
    wp_k=welder_config.rstep[0]

    # bm=128
    # bn=128
    # bk=32

    # wp_m=64
    # wp_n=64
    # wp_k=32

    row_panel=welder_config.row_panel

    # print("bs,m,n,k",bs,m,n,k)
    # print("bm,bn,bk",bm,bn,bk)
    # print("wp_m,wp_n,wp_k",wp_m,wp_n,wp_k)
    # print("mem_levels",mem_levels)
    # print("row_panel",row_panel)

    ret=calculate_ladder_matmul_resource_utilization([m,n,k],[bm,bn,bk],[wp_m,wp_n,wp_k],2,arch_spec,mem_levels,row_panel,batch=bs,split_param=1)
    post_data=hyper_post_process_tensor_core_op(ret,arch_spec, op.input1_bytes)
    
    return post_data
    

    pass
    

# # 创建操作名称到处理函数的映射
# op_handlers = {
#     # "nn_conv2d_nn_bias_add": handle_nn_conv2d_nn_bias_add,
#     # "reshape_transpose_reshape": handle_reshape_transpose_reshape,
#     "reshape": handle_reshape,
#     "transpose": handle_transpose,
#     "nn_conv2d": handle_nn_conv2d,
#     "nn_bias_add": handle_nn_bias_add,
#     "subtract": handle_subtract,
#     "multiply": handle_multiply,
#     "mean": handle_mean,
#     "add": handle_add,
#     "sqrt": handle_sqrt,
#     "divide": handle_divide,
#     "welder_C2DImplicitGemm": handle_welder_C2DImplicitGemm,
#     "strided_slice": handle_strided_slice,
#     "nn_global_avg_pool2d": handle_nn_global_avg_pool2d,
#     "nn_depth_to_space": handle_nn_depth_to_space,
#     # 其他操作处理函数
# }


def dispatch_to_modeling(op:Operation,welder_config:Welder_Config,arch_spec:Arch):
    # print("dispatch_to_modeling: op.op_name",op.op_name)
    # op.input1_level='ddr'
    # op.input2_level='ddr'
    # op.output_level='ddr'
    if op.op_name in ["expand_dims","broadcast_to"]:
        return None

    if op.op_name in['reshape', 'transpose','nn.depth_to_space',"nn.relu","layout_transform",'ladder.layout_transform',"cast",
                     "negative","concatenate","take","where","less"]: 
        # if op.input2_shape is not None:
        if op.input2_shape!=[] and op.input2_shape != [1] : # check if it is empty list
            # print("case",op.op_name,op.input2_shape)
            if (op.input1_shape!=[] or op.input1_shape != [1]):
                op.exchange_input1_input2()
            else:
                raise ValueError(f"More than 1 input in single op {op.op_name} modeling!")
        return model_element_wise_N_0(op,welder_config,arch_spec)
    if op.op_name in['sqrt','rsqrt','exp','sigmoid','tanh']: 
        # if op.input2_shape is not None:
        if op.input2_shape!=[] : # check if it is empty list
            # print("case",op.op_name,op.input2_shape)
            raise ValueError(f"More than 1 input in single op {op.op_name} modeling!")
        return model_element_wise_N_0_sfu(op,welder_config,arch_spec)
        
    elif op.op_name in ['nn.bias_add', 'subtract', 'multiply', 'add', 'divide']:
        return model_element_wise(op,welder_config,arch_spec)
    
    elif op.op_name in ['strided_slice']:
        # model_element_wise(op)
        return model_element_wise_N_0(op,welder_config,arch_spec)
    
    elif op.op_name in ['welder.C2DImplicitGemm']:
        #TODO modeling this is extremely slow
        # model_reduction(op)
        return model_conv_implicit_gemm(op,welder_config,arch_spec)
        
    elif op.op_name in ['nn.conv2d']:
        # model_reduction(op)
        return model_conv_2d(op,welder_config,arch_spec)
        pass
    elif op.op_name in ['mean', 'nn.global_avg_pool2d', "nn.avg_pool2d", "max","sum","maximum"]:
        # model_reduction(op)
        return model_reduce(op,welder_config,arch_spec)
        pass
    elif op.op_name in ['nn.matmul',"welder.matmul"]:
        return model_gemm(op,welder_config,arch_spec)
    elif op.op_name in ['ladder.perfect_matmul']:
        return model_ladder_matmul(op,welder_config,arch_spec)
    elif op.op_name in ['nn.batch_matmul']:
        return model_ladder_batch_matmul(op,welder_config,arch_spec)
    else: 
        print(op.op_name)
        raise ValueError(f"No handler found for operation: {op.op_name}")
    # 其他条件分支...

# def dispatch_to_modeling(op, reg_fused_nodes, current_relay, current_line, row_panel):
    
    
#     handler = op_handlers.get(op.op_name)
#     if handler:
#         # 接收处理函数的返回值
#         result = handler(op.op_name, reg_fused_nodes, current_relay, current_line, row_panel)
#         return result  # 将处理函数的返回值返回给调用者
#     else:
#         raise ValueError(f"No handler found for operation: {op.op_name}")
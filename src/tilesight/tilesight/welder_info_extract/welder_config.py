class Welder_Config:
    def __init__(self, reg_fused_nodes,row_panel):
        # Initialize attributes with values from reg_fused_nodes or with default values
        self.op_name = reg_fused_nodes.get('node_name', None)
        # self.grid_size = reg_fused_nodes.get('grid_size', None)

        # Initialize list attributes
        self.block = reg_fused_nodes.get('block', [])
        self.thread = reg_fused_nodes.get('thread', [])
        self.rstep = reg_fused_nodes.get('rstep', [])
        self.step = reg_fused_nodes.get('step', [])
        self.reduce_thread = reg_fused_nodes.get('reduce_thread', [])
        self.warp = reg_fused_nodes.get('warp', [])
        self.wmma = reg_fused_nodes.get('wmma', [])

        # Initialize boolean and numeric attributes
        self.use_tc = reg_fused_nodes.get('use_tc', None)
        self.use_cutlass = reg_fused_nodes.get('use_cutlass', None)

        # Initialize additional attributes with default values
        self.row_panel = row_panel
    
    def __str__(self):
        # Constructing the string representation of the object
        representation = f"op_name: {self.op_name}, use_tc: {self.use_tc}, use_cutlass: {self.use_cutlass}, row_panel: {self.row_panel}\n"
        representation += f"block: {self.block}, thread: {self.thread}, rstep: {self.rstep}, step: {self.step}, reduce_thread: {self.reduce_thread}\n"
        representation += f"warp: {self.warp}, wmma: {self.wmma}"
        return representation

    def optimized_adjust_dimensions(self,block, op_output_shape):
        import numpy as np
        from math import prod, gcd
        from functools import reduce

        # Calculate the product of the original block
        original_prod = prod(block)

        # Calculate the GCD of op_output_shape to potentially align block sizes with it
        gcd_op_shape = reduce(gcd, op_output_shape)

        # Initial attempt: distribute the gcd_op_shape as evenly as possible
        target_prod = original_prod // gcd_op_shape * gcd_op_shape
        base_value = target_prod ** (1/len(op_output_shape))

        # Initialize new block based on the base value, adjust to be less than op_output_shape dimensions
        new_block = [max(1, min(int(base_value), shape - 1)) for shape in op_output_shape]

        # Function to adjust the block to get as close as possible to the original product while ensuring divisibility
        def adjust_block(block, op_shape, target_prod):
            current_prod = prod(block)
            for i, val in enumerate(block):
                while current_prod < target_prod and block[i] < op_shape[i] - 1:
                    block[i] += 1
                    current_prod = prod(block)
                if current_prod >= target_prod:
                    break
            return block

        # Adjust the new block to try and match the original product while adhering to the new constraints
        adjusted_block = adjust_block(new_block, op_output_shape, original_prod)

        return adjusted_block

    def get_block_thread(self,op_output_shape):

        # thread_block
        if len(self.block)!=len(op_output_shape):
            current_block=self.optimized_adjust_dimensions(self.block, op_output_shape)
            print("Output shape",op_output_shape, "and Thread Block Tile dimension", self.block ,"mismatch at",self.op_name,"kenerl! Now adjusted.")
        else:
            current_block=self.block

        # dim_threads
        if self.thread==[]:
            current_dim_threads=self.optimized_adjust_dimensions([128], current_block)
        elif len(self.thread)!=len(op_output_shape):
            current_dim_threads=self.optimized_adjust_dimensions(self.thread, current_block)
            print("Dim Threads",self.thread, "and Thread Block Tile dimension", current_block ,"mismatch at",self.op_name,"kenerl! Now adjusted.")
        else:
            current_dim_threads=self.thread

        

        return current_block,current_dim_threads
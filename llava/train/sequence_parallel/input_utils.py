import torch


def extract_local_from_list(vaule_list, sp_rank, sp_size):
    quotient, remainder = divmod(len(vaule_list), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    return vaule_list[start_idx:end_idx]


def extract_local_input_ids(input_ids, image_positions, sp_rank, sp_size, bos_token_id=1, image_token_len=3):
    quotient, remainder = divmod(len(image_positions), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)

    start_position_idx = image_positions[start_idx]
    if sp_rank != sp_size - 1:
        end_position_idx = image_positions[end_idx]
    else:
        end_position_idx = len(input_ids)

    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]


def extract_local_position_ids(input_ids, image_positions, image_ids, sp_rank, sp_size, image_token_len=198):
    quotient, remainder = divmod(len(image_ids), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    start_position_idx = image_positions[start_idx] + image_ids[start_idx] * image_token_len
    if sp_rank != sp_size - 1:  # Handle the tail of the sequence
        end_position_idx = image_positions[end_idx] + image_ids[end_idx] * image_token_len  # image_token_len + 3
    else:
        end_position_idx = len(input_ids)
    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]


def extract_local(value, rank, world_size, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value


def prepare_hybrid_attn_inputs(input_ids, position_ids, target_ids, rank, world_size, device):
    local_input_ids = extract_local(
        input_ids,
        rank,
        world_size,
        device,
    )
    local_position_ids = extract_local(
        position_ids,
        rank,
        world_size,
        device,
    )
    if target_ids is not None:
        local_target_ids = extract_local(
            target_ids,
            rank,
            world_size,
            device,
        )
    else:
        local_target_ids = None
    return {
        "local_input_ids": local_input_ids,
        "local_position_ids": local_position_ids,
        "local_target_ids": local_target_ids,
    }

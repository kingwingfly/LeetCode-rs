// 我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。
// 同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。
// 这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。
#[cfg(feature = "31_60")]
fn next_permutation(nums: &mut Vec<i32>) {
    for l in (0..nums.len() - 1).rev() {
        if nums[l] < nums[l + 1] {
            let mut r = l + 1;
            while r + 1 < nums.len() && nums[r + 1] > nums[l] {
                r += 1;
            }
            // (nums[l], nums[r]) = (nums[r], nums[l]);
            nums[l] = nums[l] ^ nums[r];
            nums[r] = nums[l] ^ nums[r];
            nums[l] = nums[l] ^ nums[r];
            let (_, r) = nums.split_at_mut(l + 1);
            r.sort();
            break;
        }
    }
    nums.sort();
}

#[cfg(feature = "31_60")]
fn longest_valid_parentheses(s: String) -> i32 {
    let mut dp = vec![0; s.len()];
    let s = s.as_bytes();
    let mut ret = 0;
    for i in 1..s.len() {
        match s[i] {
            b')' => match s[i - 1] {
                b'(' => {
                    dp[i] = 2;
                    if i > 2 {
                        dp[i] += dp[i - 2];
                    }
                }
                b')' if dp[i - 1] > 0 => {
                    if i - dp[i - 1] >= 1 && s[i - dp[i - 1] - 1] == b'(' {
                        dp[i] = dp[i - 1] + 2;
                        if i - dp[i - 1] >= 2 {
                            dp[i] = dp[i] + dp[i - dp[i - 1] - 2];
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        ret = ret.max(dp[i]);
    }
    ret as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

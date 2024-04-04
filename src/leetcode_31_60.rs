// 我们需要将一个左边的「较小数」与一个右边的「较大数」交换，以能够让当前排列变大，从而得到下一个排列。
// 同时我们要让这个「较小数」尽量靠右，而「较大数」尽可能小。当交换完成后，「较大数」右边的数需要按照升序重新排列。
// 这样可以在保证新排列大于原来排列的情况下，使变大的幅度尽可能小。
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

fn search(nums: Vec<i32>, target: i32) -> i32 {
    let (mut l, mut r) = (0, nums.len() - 1);
    while l < r {
        let mid = (l + r) >> 1;
        if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    match nums.get(l) {
        Some(&x) if x == target && l == r => l as i32,
        _ => -1,
    }
}

fn search_range(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let (l, r) = (
        nums.partition_point(|&x| x < target),
        nums.partition_point(|&x| x <= target),
    );
    if l < nums.len() && nums[l] == target && r >= 1 {
        vec![l as i32, (r - 1) as i32]
    } else {
        vec![-1, -1]
    }
}

fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    nums.partition_point(|&x| x < target) as i32
}

mod sudoku {
    pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
        let (mut rows, mut cols, mut blocks) = ([0; 9], [0; 9], [0; 9]);
        for i in 0..9 {
            for j in 0..9 {
                let c = board[i][j];
                if c == '.' {
                    continue;
                }
                let n = c.to_digit(10).unwrap();
                let block_id = i / 3 * 3 + j / 3;
                if ((rows[i] >> n) & 1 | (cols[j] >> n) & 1 | (blocks[block_id] >> n) & 1) == 1 {
                    return false;
                }
                rows[i] |= 1 << n;
                cols[j] |= 1 << n;
                blocks[block_id] |= 1 << n;
            }
        }
        true
    }

    type Board = Vec<Vec<char>>;

    #[derive(Default)]
    pub struct SudokuSolution {
        pub board: Board,
        rows: [u16; 9],
        cols: [u16; 9],
        blocks: [u16; 9],
    }

    impl SudokuSolution {
        pub fn new(board: Board) -> Self {
            Self {
                board,
                ..Default::default()
            }
        }

        pub fn solve_sudoku(&mut self) {
            let mut cnt = 0;
            for i in 0..9 {
                for j in 0..9 {
                    let c = self.board[i][j];
                    if c == '.' {
                        cnt += 1;
                        continue;
                    }
                    let n = c.to_digit(10).unwrap();
                    let block_id = i / 3 * 3 + j / 3;
                    self.rows[i] |= 1 << n;
                    self.cols[j] |= 1 << n;
                    self.blocks[block_id] |= 1 << n;
                }
            }
            self.dfs(cnt);
        }

        fn dfs(&mut self, cnt: usize) -> bool {
            if cnt == 0 {
                return true;
            }
            let (i, j, bits) = self.get_next();
            for n in 1..10 {
                if bits & (1 << n) == 0 {
                    continue;
                }
                let n = std::char::from_digit(n, 10).unwrap();
                self.fill_num(i, j, n);
                if self.dfs(cnt - 1) {
                    return true;
                }
                self.cancel_fill_num(i, j, n);
            }
            false
        }

        fn get_possible(&self, i: usize, j: usize) -> u16 {
            !(self.rows[i] | self.cols[j] | self.blocks[i / 3 * 3 + j / 3])
        }

        fn get_next(&self) -> (usize, usize, u16) {
            let (mut x, mut y) = (0, 0);
            let mut min = 10;
            let mut bits_ret = 0b1_111_111_111;
            for i in 0..9 {
                for j in 0..9 {
                    if self.board[i][j] != '.' {
                        continue;
                    }
                    let bits = self.get_possible(i, j);
                    let possible_num = bits.count_ones() as u8;
                    if possible_num >= min {
                        continue;
                    }
                    min = possible_num;
                    (x, y) = (i, j);
                    bits_ret = bits;
                }
            }
            (x, y, bits_ret)
        }

        fn fill_num(&mut self, i: usize, j: usize, n: char) {
            self.board[i][j] = n;
            let n = n.to_digit(10).unwrap() as u16;
            self.rows[i] |= 1 << n;
            self.cols[j] |= 1 << n;
            self.blocks[i / 3 * 3 + j / 3] |= 1 << n;
        }

        fn cancel_fill_num(&mut self, i: usize, j: usize, n: char) {
            self.board[i][j] = '.';
            let n = n.to_digit(10).unwrap() as u16;
            self.rows[i] ^= 1 << n;
            self.cols[j] ^= 1 << n;
            self.blocks[i / 3 * 3 + j / 3] ^= 1 << n;
        }
    }
}

fn count_and_say(n: i32) -> String {
    let mut s = "1".to_string();
    for _ in 1..n {
        let mut t = String::new();
        let mut i = 0;
        while i < s.len() {
            let mut j = i + 1;
            while j < s.len() && s.as_bytes()[i] == s.as_bytes()[j] {
                j += 1;
            }
            t.push_str(&(j - i).to_string());
            t.push(s.as_bytes()[i] as char);
            i = j;
        }
        s = t;
    }
    s
}

fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(
        a: &Vec<i32>,
        cur: i32,
        target: i32,
        idx: usize,
        tmp: &mut Vec<i32>,
        ans: &mut Vec<Vec<i32>>,
    ) {
        match cur.cmp(&target) {
            std::cmp::Ordering::Less => {
                for i in idx..a.len() {
                    tmp.push(a[i]);
                    dfs(a, cur + a[i], target, i, tmp, ans);
                    tmp.pop();
                }
            }
            std::cmp::Ordering::Equal => {
                ans.push(tmp.clone());
            }
            std::cmp::Ordering::Greater => {}
        }
    }
    let mut ans = Vec::with_capacity(150);
    dfs(
        &candidates,
        0,
        target,
        0,
        &mut Vec::with_capacity(64),
        &mut ans,
    );
    ans
}

fn combination_sum2(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(
        a: &Vec<i32>,
        cur: i32,
        target: i32,
        idx: usize,
        tmp: &mut Vec<i32>,
        ans: &mut Vec<Vec<i32>>,
    ) {
        match cur.cmp(&target) {
            std::cmp::Ordering::Less => {
                for i in idx..a.len() {
                    if i > idx && a[i] == a[i - 1] {
                        continue;
                    }
                    tmp.push(a[i]);
                    dfs(a, cur + a[i], target, i + 1, tmp, ans);
                    tmp.pop();
                }
            }
            std::cmp::Ordering::Equal => {
                ans.push(tmp.clone());
            }
            std::cmp::Ordering::Greater => {}
        }
    }
    candidates.sort_unstable();
    let mut ans = Vec::with_capacity(150);
    dfs(
        &candidates,
        0,
        target,
        0,
        &mut Vec::with_capacity(64),
        &mut ans,
    );
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let ans = combination_sum2(vec![2, 5, 2, 1, 2], 5);
        dbg!(ans);
    }
}

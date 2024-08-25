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

fn first_missing_positive(mut nums: Vec<i32>) -> i32 {
    let l = nums.len();
    for i in nums.iter_mut() {
        if *i <= 0 {
            *i = l as i32 + 1;
        }
    }
    for i in 0..l {
        let n = nums[i].unsigned_abs() as usize;
        if n <= l {
            nums[n - 1] = -nums[n - 1].abs();
        }
    }
    for (i, n) in nums.iter().enumerate() {
        if *n > 0 {
            return i as i32 + 1;
        }
    }
    nums.len() as i32 + 1
}

fn trap(height: Vec<i32>) -> i32 {
    let (mut l, mut r) = (0, height.len() - 1);
    let (mut l_max, mut r_max) = (0, 0);
    let mut ans = 0;
    while l < r {
        if height[l] < height[r] {
            if height[l] < l_max {
                ans += l_max - height[l];
            } else {
                l_max = height[l];
            }
            l += 1;
        } else {
            if height[r] < r_max {
                ans += r_max - height[r];
            } else {
                r_max = height[r];
            }
            r -= 1;
        }
    }
    ans
}

fn multiply(num1: String, num2: String) -> String {
    if num1 == "0" || num2 == "0" {
        return "0".to_string();
    }
    let mut mul: Vec<i32> = vec![0; num1.len() + num2.len()];
    let c1: Vec<i32> = num1.chars().rev().map(|x| x as i32 - 48).collect();
    let c2: Vec<i32> = num2.chars().rev().map(|x| x as i32 - 48).collect();
    for i in 0..c1.len() {
        for j in 0..c2.len() {
            mul[i + j] += c1[i] * c2[j];
        }
    }
    for i in 0..mul.len() - 1 {
        mul[i + 1] += mul[i] / 10;
        mul[i] %= 10;
    }
    mul.into_iter()
        .rev()
        .skip_while(|&x| x == 0)
        .fold(String::new(), |mut s, x| {
            s.push((x + 48) as u8 as char);
            s
        })
}

fn is_match(s: String, p: String) -> bool {
    let (n, m) = (s.len(), p.len());
    let new_s = " ".to_owned() + &s;
    let new_p = " ".to_owned() + &p;

    let mut dp = vec![vec![false; m + 1]; n + 1];
    dp[0][0] = true;

    for (i, c1) in new_s.chars().enumerate() {
        for (j, c2) in new_p.chars().enumerate().skip(1) {
            if c2 == '*' {
                dp[i][j] = dp[i][j - 1] || (i >= 1 && dp[i - 1][j]);
            } else {
                dp[i][j] = i >= 1 && dp[i - 1][j - 1] && (c1 == c2 || c2 == '?');
            }
        }
    }
    dp[n][m]
}

fn is_match2(s: String, p: String) -> bool {
    let s = s.as_bytes();
    let p = p.as_bytes();
    let (mut i, mut j, mut m) = (0, 0, 0);
    let mut start = None;

    while i < s.len() {
        if j < p.len() && (s[i] == p[j] || p[j] == b'?') {
            i += 1;
            j += 1;
        } else if j < p.len() && p[j] == b'*' {
            start = Some(j);
            m = i; // 记录*匹配的位置
            j += 1;
        } else if let Some(start) = start {
            // 如果遇到不匹配的字符，回溯到m+1的位置，重新匹配
            j = start + 1;
            m += 1;
            i = m;
        } else {
            return false;
        }
    }

    while j < p.len() {
        if p[j] != b'*' {
            return false;
        }
        j += 1
    }
    true
}

fn jump(nums: Vec<i32>) -> i32 {
    use std::cmp::max;
    let mut steps = 0;
    let mut end = 0;
    let mut max_pos = 0;
    for (i, n) in nums[..nums.len() - 1].iter().enumerate() {
        max_pos = max(max_pos, n + i as i32); // 记录当前位置能跳到的最远位置
        if i as i32 == end {
            // 如果当前位置等于上一次跳跃的最远位置，说明需要跳一次
            end = max_pos; // 更新最远位置
            steps += 1;
        }
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let ans = jump(vec![2, 3, 1, 1, 4]);
        dbg!(ans);
    }
}

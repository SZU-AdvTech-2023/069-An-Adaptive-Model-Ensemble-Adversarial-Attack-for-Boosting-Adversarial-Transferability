class Flowshop {
   friend Flow(int**, int, int []);
   private:
      void Backtrack(int i);
      int   **M,    // 各作业所需的处理时间
            *x,     // 当前作业调度
            *bestx, // 当前最优作业调度
            *f2,    // 机器2完成处理时间
            f1,     // 机器1完成处理时间
            f,      // 完成时间和
            bestf,  // 当前最优值
            n;      // 作业数}; 
